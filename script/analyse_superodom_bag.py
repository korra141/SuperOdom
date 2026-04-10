#!/usr/bin/env python3
"""
Offline analysis and plotting of super_odometry bag recordings.

Reads a rosbag2 recorded by record_superodom.sh and produces:
  - Trajectory plots (top-down XY, XYZ vs time)
  - Orientation (roll/pitch/yaw) vs time
  - Optimisation stats (feature counts, latency, iteration counts, distances)
  - Uncertainty / confidence metrics per axis
  - Feature-match rejection breakdown
  - Prediction source timeline
  - IMU vs LiDAR odometry comparison
  - Inter-source drift via RPE (evo) — LiDAR vs IMU at matched timestamps
  - Map consistency via submap ICP + optional MME (--map-consistency, requires open3d)

Usage:
    python3 analyse_superodom_bag.py -b <bag_dir> [-o <output_dir>] [-p <prefix>]

Dependencies:
    pip install rosbag2-py rclpy matplotlib numpy scipy evo
    (rclpy needed only for message deserialisation; install via ROS2 overlay)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Optional evo dependency (segment consistency metric)
# ---------------------------------------------------------------------------

try:
    from evo.core import metrics, trajectory, sync
    from evo.core.metrics import PoseRelation, Unit
    HAS_EVO = True
except ImportError:
    HAS_EVO = False
    print("[WARN] evo not found — inter-source drift metric will be skipped. "
          "Install with: pip install evo")

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    # Only warn if --map-consistency is actually requested (checked at runtime)


# ---------------------------------------------------------------------------
# rosbag2 reading helpers
# ---------------------------------------------------------------------------

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[WARN] rosbag2_py / rclpy not found — run inside a sourced ROS2 workspace.")
    sys.exit(1)


def open_bag(bag_path: str):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def topic_type_map(reader) -> dict:
    return {info.name: info.type for info in reader.get_all_topics_and_types()}


def read_messages(bag_path: str, topics: list[str]) -> dict[str, list]:
    """Return {topic: [(stamp_sec, msg), ...]} for the requested topics."""
    reader = open_bag(bag_path)
    types = topic_type_map(reader)

    filter_ = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(filter_)

    msg_classes = {}
    for t in topics:
        if t in types:
            try:
                msg_classes[t] = get_message(types[t])
            except Exception as e:
                print(f"[WARN] cannot load message type '{types[t]}' for topic '{t}': {e}")

    data: dict[str, list] = {t: [] for t in topics}
    while reader.has_next():
        topic, raw, stamp_ns = reader.read_next()
        if topic not in msg_classes:
            continue
        msg = deserialize_message(raw, msg_classes[topic])
        stamp_sec = stamp_ns * 1e-9
        # prefer header stamp when available
        if hasattr(msg, "header"):
            s = msg.header.stamp
            stamp_sec = s.sec + s.nanosec * 1e-9
        data[topic].append((stamp_sec, msg))

    return data


# ---------------------------------------------------------------------------
# Reactive bag processor — publish / subscribe over a single bag pass
# ---------------------------------------------------------------------------

class BagProcessor:
    """
    Streams a rosbag2 file once, routing each deserialized message to every
    subscriber registered for that topic.  Subscribers are plain callables:
        handler(stamp_sec: float, msg) -> None
    Only topics that have at least one subscriber are read.
    """

    def __init__(self, bag_path: str):
        self._bag_path = bag_path
        self._handlers: dict[str, list] = {}

    def subscribe(self, topic: str, handler) -> None:
        self._handlers.setdefault(topic, []).append(handler)

    def spin(self) -> None:
        topics = list(self._handlers.keys())
        if not topics:
            return

        reader = open_bag(self._bag_path)
        types  = topic_type_map(reader)

        msg_classes = {}
        for t in topics:
            if t in types:
                try:
                    msg_classes[t] = get_message(types[t])
                except Exception as e:
                    print(f"[WARN] cannot load type for '{t}': {e}")

        reader.set_filter(rosbag2_py.StorageFilter(topics=list(msg_classes.keys())))

        while reader.has_next():
            topic, raw, stamp_ns = reader.read_next()
            if topic not in msg_classes:
                continue
            msg = deserialize_message(raw, msg_classes[topic])
            stamp_sec = stamp_ns * 1e-9
            if hasattr(msg, "header"):
                s = msg.header.stamp
                stamp_sec = s.sec + s.nanosec * 1e-9
            for handler in self._handlers.get(topic, []):
                handler(stamp_sec, msg)


class OdomSubscriber:
    """Extracts pose scalars on the fly; never stores raw messages."""

    def __init__(self):
        self._t_abs: list[float] = []
        self._x: list[float] = []
        self._y: list[float] = []
        self._z: list[float] = []
        self._roll: list[float] = []
        self._pitch: list[float] = []
        self._yaw: list[float] = []

    @property
    def t0_abs(self) -> float:
        return self._t_abs[0] if self._t_abs else 0.0

    def __call__(self, stamp: float, msg) -> None:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        r, p_, yw = _quat_to_rpy_raw(o.x, o.y, o.z, o.w)
        self._t_abs.append(stamp)
        self._x.append(p.x); self._y.append(p.y); self._z.append(p.z)
        self._roll.append(math.degrees(r))
        self._pitch.append(math.degrees(p_))
        self._yaw.append(math.degrees(yw))

    def result(self) -> dict:
        if not self._t_abs:
            return {}
        t0 = self._t_abs[0]
        return dict(
            t=np.array(self._t_abs) - t0,
            x=np.array(self._x), y=np.array(self._y), z=np.array(self._z),
            roll=np.array(self._roll), pitch=np.array(self._pitch), yaw=np.array(self._yaw),
        )


class StatsSubscriber:
    """Stores (stamp, msg) for optimisation stats — messages are small scalars."""

    def __init__(self):
        self.data: list = []

    def __call__(self, stamp: float, msg) -> None:
        self.data.append((stamp, msg))


class PathSubscriber:
    """Keeps only the last Path message to avoid accumulating all poses."""

    def __init__(self):
        self._last = None

    def __call__(self, stamp: float, msg) -> None:
        self._last = msg

    def result(self) -> tuple:
        if self._last is None:
            return np.array([]), np.array([]), np.array([])
        xs = [p.pose.position.x for p in self._last.poses]
        ys = [p.pose.position.y for p in self._last.poses]
        zs = [p.pose.position.z for p in self._last.poses]
        return np.array(xs), np.array(ys), np.array(zs)


class FeatureGeometrySubscriber:
    """
    Processes /feature_info messages on the fly: extracts per-frame geometry
    stats (point counts + eigenvalue ratio) and discards the raw point clouds
    immediately after each message is handled.
    """

    _CLOUD_FIELDS = {
        "nodistortion": "cloud_nodistortion",
        "corner":       "cloud_corner",
        "surface":      "cloud_surface",
        "realsense":    "cloud_realsense",
    }

    def __init__(self):
        self._t0: float | None = None
        self._clouds: dict[str, tuple] = {k: ([], [], [], []) for k in self._CLOUD_FIELDS}
        self._combined: tuple = ([], [], [], [])

    def __call__(self, stamp: float, msg) -> None:
        if self._t0 is None:
            self._t0 = stamp
        t_rel = stamp - self._t0
        frame_feature_pts = []

        for key, field in self._CLOUD_FIELDS.items():
            t_list, cnt_list, ratio_list, emin_list = self._clouds[key]
            pc2_msg = getattr(msg, field, None)
            pts = _pc2_msg_to_xyz(pc2_msg) if pc2_msg is not None else np.empty((0, 3))
            ratio, emin = _geometry_stats(pts)
            t_list.append(t_rel)
            cnt_list.append(len(pts))
            ratio_list.append(ratio)
            emin_list.append(emin)
            if key != "nodistortion" and len(pts) > 0:
                frame_feature_pts.append(pts)

        t_c, cnt_c, ratio_c, emin_c = self._combined
        if frame_feature_pts:
            merged = np.vstack(frame_feature_pts)
            c_ratio, c_emin = _geometry_stats(merged)
            t_c.append(t_rel); cnt_c.append(len(merged))
            ratio_c.append(c_ratio); emin_c.append(c_emin)
        else:
            t_c.append(t_rel); cnt_c.append(0)
            ratio_c.append(float("nan")); emin_c.append(float("nan"))

    def result(self) -> dict:
        out = {}
        for key, (t_list, cnt_list, ratio_list, emin_list) in self._clouds.items():
            arr_cnt = np.array(cnt_list, dtype=float)
            if arr_cnt.sum() == 0:
                continue
            out[key] = dict(
                t=np.array(t_list), count=arr_cnt,
                eig_ratio=np.array(ratio_list), eig_min=np.array(emin_list),
            )
        t_c, cnt_c, ratio_c, emin_c = self._combined
        arr_cnt_c = np.array(cnt_c, dtype=float)
        if arr_cnt_c.sum() > 0:
            out["combined"] = dict(
                t=np.array(t_c), count=arr_cnt_c,
                eig_ratio=np.array(ratio_c), eig_min=np.array(emin_c),
            )
        return out


class SubmapBuilderSubscriber:
    """
    Assigns each incoming registered_scan to its submap bucket by absolute
    timestamp, voxel-downsampling on the fly to keep memory bounded.
    Boundaries are absolute ROS timestamps (seconds).
    """

    def __init__(self, boundaries_abs: list[float], voxel_size: float = 0.1):
        self._boundaries = boundaries_abs
        self._voxel = voxel_size
        self._n = len(boundaries_abs) - 1
        # Each bucket holds already-downsampled chunks
        self._buckets: list[list[np.ndarray]] = [[] for _ in range(self._n)]
        self._frame_counts = [0] * self._n
        self._FLUSH_EVERY = 50   # voxel-downsample every N frames per bucket

    def __call__(self, stamp: float, msg) -> None:
        for i in range(self._n):
            if self._boundaries[i] <= stamp < self._boundaries[i + 1]:
                xyz = _pc2_to_xyz(msg)
                if xyz.size:
                    self._buckets[i].append(xyz)
                    self._frame_counts[i] += 1
                    if self._frame_counts[i] % self._FLUSH_EVERY == 0:
                        self._flush(i)
                break

    def _flush(self, i: int) -> None:
        """Merge and voxel-downsample bucket i to shed raw-point memory."""
        if not self._buckets[i]:
            return
        merged = np.vstack(self._buckets[i])
        if HAS_O3D:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
            pcd = pcd.voxel_down_sample(self._voxel)
            merged = np.asarray(pcd.points, dtype=np.float32)
        self._buckets[i] = [merged]

    def result(self) -> list[np.ndarray]:
        """Return one merged+downsampled array per submap."""
        out = []
        for i in range(self._n):
            self._flush(i)
            if self._buckets[i]:
                out.append(np.vstack(self._buckets[i]).astype(np.float32))
            else:
                out.append(np.empty((0, 3), dtype=np.float32))
        return out


class LastMessageSubscriber:
    """Keeps only the last message received on a topic."""

    def __init__(self):
        self.stamp: float | None = None
        self.msg = None

    def __call__(self, stamp: float, msg) -> None:
        self.stamp = stamp
        self.msg = msg


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_to_rpy_raw(qx, qy, qz, qw):
    """Convert quaternion to roll/pitch/yaw (radians)."""
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def extract_odom(data: list) -> dict:
    """Pull arrays from a list of (stamp, Odometry) tuples."""
    if not data:
        return {}
    t, x, y, z, roll, pitch, yaw = [], [], [], [], [], [], []
    for stamp, msg in data:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        r, p_, yw = _quat_to_rpy_raw(o.x, o.y, o.z, o.w)
        t.append(stamp); x.append(p.x); y.append(p.y); z.append(p.z)
        roll.append(math.degrees(r)); pitch.append(math.degrees(p_)); yaw.append(math.degrees(yw))
    t0 = t[0]
    return dict(t=np.array(t) - t0, x=np.array(x), y=np.array(y), z=np.array(z),
                roll=np.array(roll), pitch=np.array(pitch), yaw=np.array(yaw))


def extract_path(data: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) arrays from the last Path message received."""
    if not data:
        return np.array([]), np.array([]), np.array([])
    _, msg = data[-1]
    xs = [p.pose.position.x for p in msg.poses]
    ys = [p.pose.position.y for p in msg.poses]
    zs = [p.pose.position.z for p in msg.poses]
    return np.array(xs), np.array(ys), np.array(zs)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_trajectory(odoms: dict[str, dict], path_xy: tuple, out: Path):
    """Top-down XY trajectory from odometry and path topics."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, od), c in zip(odoms.items(), colors):
        if od:
            ax.plot(od["x"], od["y"], "-", color=c, linewidth=0.8, label=label)
            ax.plot(od["x"][0], od["y"][0], "o", color=c, markersize=5)

    px, py, _ = path_xy
    if px.size:
        ax.plot(px, py, "--", color="black", linewidth=0.6, alpha=0.5, label="laser_odom_path")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory — top-down (XY)")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.4)
    _save(fig, out / "trajectory_xy.png")


def plot_xyz_vs_time(odoms: dict[str, dict], out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, od), c in zip(odoms.items(), colors):
        if not od:
            continue
        for ax, key, ylabel in zip(axes, ["x", "y", "z"], ["X (m)", "Y (m)", "Z (m)"]):
            ax.plot(od["t"], od[key], "-", color=c, linewidth=0.8, label=label)
    for ax, ylabel in zip(axes, ["X (m)", "Y (m)", "Z (m)"]):
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.4); ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position vs Time")
    plt.tight_layout()
    _save(fig, out / "position_vs_time.png")


def plot_rpy_vs_time(odoms: dict[str, dict], out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, od), c in zip(odoms.items(), colors):
        if not od:
            continue
        for ax, key, ylabel in zip(axes, ["roll", "pitch", "yaw"],
                                    ["Roll (°)", "Pitch (°)", "Yaw (°)"]):
            ax.plot(od["t"], od[key], "-", color=c, linewidth=0.8, label=label)
    for ax, ylabel in zip(axes, ["Roll (°)", "Pitch (°)", "Yaw (°)"]):
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.4); ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Orientation (RPY) vs Time")
    plt.tight_layout()
    _save(fig, out / "orientation_vs_time.png")


def plot_optimisation_stats(stats_data: list, out: Path):
    if not stats_data:
        print("  [skip] no optimisation stats messages")
        return

    t0 = stats_data[0][0]
    t = np.array([s - t0 for s, _ in stats_data])
    msgs = [m for _, m in stats_data]

    def col(field):
        return np.array([getattr(m, field) for m in msgs])

    # --- feature counts ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    pairs = [
        ("laser_cloud_surf_from_map_num",    "Surf from map",    "tab:green"),
        ("laser_cloud_corner_from_map_num",  "Corner from map",  "tab:blue"),
        ("laser_cloud_surf_stack_num",       "Surf stack",       "gold"),
        ("laser_cloud_corner_stack_num",     "Corner stack",     "magenta"),
    ]
    for ax, (field, label, c) in zip(axes.flat, pairs):
        ax.plot(t, col(field), "-", color=c, linewidth=0.8)
        ax.set_title(label, fontsize=10); ax.set_ylabel("count")
        ax.grid(True, alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Feature Counts")
    plt.tight_layout()
    _save(fig, out / "stats_feature_counts.png")

    # --- performance ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (field, label, c) in zip(axes, [
        ("latency",     "Latency (ms)",          "red"),
        ("time_elapsed","Time elapsed (s)",       "teal"),
        ("n_iterations","Optimiser iterations",   "darkgreen"),
    ]):
        ax.plot(t, col(field), "-", color=c, linewidth=0.8)
        ax.set_ylabel(label); ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Performance Metrics")
    plt.tight_layout()
    _save(fig, out / "stats_performance.png")

    # --- displacement ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (field, label, c) in zip(axes, [
        ("total_translation",    "Total translation (m)", "orange"),
        ("translation_from_last","Δ translation (m)",     "orangered"),
        ("average_distance",     "Avg feature distance (m)", "steelblue"),
    ]):
        ax.plot(t, col(field), "-", color=c, linewidth=0.8)
        ax.set_ylabel(label); ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Displacement Metrics")
    plt.tight_layout()
    _save(fig, out / "stats_displacement.png")

    # --- uncertainty / confidence ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    unc_fields = [
        ("uncertainty_x",     "Unc X",     "tab:blue"),
        ("uncertainty_y",     "Unc Y",     "tab:orange"),
        ("uncertainty_z",     "Unc Z",     "tab:green"),
        ("uncertainty_roll",  "Unc Roll",  "tab:red"),
        ("uncertainty_pitch", "Unc Pitch", "tab:purple"),
        ("uncertainty_yaw",   "Unc Yaw",   "tab:brown"),
    ]
    for ax, (field, label, c) in zip(axes.flat, unc_fields):
        ax.plot(t, col(field), "-", color=c, linewidth=0.8)
        ax.set_title(label, fontsize=9); ax.grid(True, alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Pose Uncertainty (Confidence)")
    plt.tight_layout()
    _save(fig, out / "stats_uncertainty.png")

    # --- plane match rejection breakdown (stacked area) ---
    rejection_fields = [
        ("plane_match_success",       "Success",          "tab:green"),
        ("plane_no_enough_neighbor",  "No neighbour",     "tab:orange"),
        ("plane_neighbor_too_far",    "Neighbour too far","tab:red"),
        ("plane_badpca_structure",    "Bad PCA",          "tab:purple"),
        ("plane_invalid_numerical",   "Invalid numerical","tab:brown"),
        ("plane_mse_too_large",       "MSE too large",    "tab:pink"),
        ("plane_unknown",             "Unknown",          "gray"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(t))
    for field, label, c in rejection_fields:
        vals = col(field).astype(float)
        ax.fill_between(t, bottom, bottom + vals, label=label, alpha=0.8, color=c)
        bottom += vals
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Point count")
    ax.set_title("Plane Feature Match Outcome Breakdown")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)
    _save(fig, out / "stats_plane_rejection.png")

    # --- prediction source ---
    PRED_LABELS = {0: "IMU_ORIENT", 1: "LIO_ODOM", 2: "VIO_ODOM", 3: "NEURAL_IMU", 4: "CONST_VEL"}
    src = col("prediction_source")
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.scatter(t, src, s=4, c=src, cmap="tab10", vmin=0, vmax=4)
    ax.set_yticks(list(PRED_LABELS.keys()))
    ax.set_yticklabels(list(PRED_LABELS.values()), fontsize=8)
    ax.set_xlabel("Time (s)"); ax.set_title("Active Prediction Source")
    ax.grid(True, alpha=0.3)
    _save(fig, out / "stats_prediction_source.png")


def plot_rotation_vs_time(stats_data: list, out: Path):
    if not stats_data:
        return
    t0 = stats_data[0][0]
    t = np.array([s - t0 for s, _ in stats_data])
    total_rot = np.array([math.degrees(m.total_rotation) for _, m in stats_data])
    delta_rot = np.array([math.degrees(m.rotation_from_last) for _, m in stats_data])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, total_rot, "-", color="purple", linewidth=0.8)
    axes[0].set_ylabel("Total rotation (°)"); axes[0].grid(True, alpha=0.4)
    axes[1].plot(t, delta_rot, "-", color="indigo", linewidth=0.8)
    axes[1].set_ylabel("Δ rotation (°)"); axes[1].grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Rotation Metrics")
    plt.tight_layout()
    _save(fig, out / "stats_rotation.png")


def plot_residuals(stats_data: list, out: Path):
    """Plot point-to-line (edge) and point-to-plane (surface) RMS residuals vs time."""
    if not stats_data:
        print("  [skip] residuals — no optimisation stats messages")
        return

    t0 = stats_data[0][0]
    t = np.array([s - t0 for s, _ in stats_data])
    msgs = [m for _, m in stats_data]

    edge_rms  = np.array([getattr(m, "edge_residual_rms",  float("nan")) for m in msgs])
    plane_rms = np.array([getattr(m, "plane_residual_rms", float("nan")) for m in msgs])

    if np.all(np.isnan(edge_rms)) and np.all(np.isnan(plane_rms)):
        print("  [skip] residuals — fields not present in bag (rebuild required)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(t, edge_rms * 1e3, "-", color="tab:blue", linewidth=0.8)
    axes[0].set_ylabel("Point-to-line RMS (mm)")
    axes[0].set_title("Edge (point-to-line) residual")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t, plane_rms * 1e3, "-", color="tab:orange", linewidth=0.8)
    axes[1].set_ylabel("Point-to-plane RMS (mm)")
    axes[1].set_title("Surface (point-to-plane) residual")
    axes[1].grid(True, alpha=0.4)
    axes[1].set_xlabel("Time (s)")

    fig.suptitle("Optimisation Residuals (final pose, per scan)")
    plt.tight_layout()
    _save(fig, out / "residuals.png")


def plot_association_distances(stats_data: list, out: Path):
    """Plot per-feature-type association distance distributions per scan.

    assoc_distance = sqrt(nearest_dist[0]) at match time, before optimisation.
    A large mean or heavy tail indicates wrong/far correspondences.
    Comparing edge vs plane reveals which feature type has worse data association.
    """
    if not stats_data:
        print("  [skip] association distances — no optimisation stats messages")
        return

    t0   = stats_data[0][0]
    t    = np.array([s - t0 for s, _ in stats_data])
    msgs = [m for _, m in stats_data]

    edge_mean  = np.array([getattr(m, "edge_assoc_dist_mean",  float("nan")) for m in msgs])
    edge_max   = np.array([getattr(m, "edge_assoc_dist_max",   float("nan")) for m in msgs])
    plane_mean = np.array([getattr(m, "plane_assoc_dist_mean", float("nan")) for m in msgs])
    plane_max  = np.array([getattr(m, "plane_assoc_dist_max",  float("nan")) for m in msgs])

    if np.all(np.isnan(edge_mean)) and np.all(np.isnan(plane_mean)):
        print("  [skip] association distances — fields not present in bag (rebuild required)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Top: mean association distance per scan
    axes[0].plot(t, edge_mean  * 1e2, "-", color="tab:blue",   linewidth=0.9, label="edge mean")
    axes[0].plot(t, plane_mean * 1e2, "-", color="tab:orange", linewidth=0.9, label="plane mean")
    axes[0].set_ylabel("Mean assoc. distance (cm)")
    axes[0].set_title("Mean nearest-map-point distance at match time  (lower = tighter association)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)

    # Bottom: max association distance per scan (sensitivity to wrong matches)
    axes[1].plot(t, edge_max  * 1e2, "-", color="tab:blue",   linewidth=0.8, alpha=0.8, label="edge max")
    axes[1].plot(t, plane_max * 1e2, "-", color="tab:orange", linewidth=0.8, alpha=0.8, label="plane max")
    axes[1].set_ylabel("Max assoc. distance (cm)")
    axes[1].set_title("Worst-case association distance per scan  (spikes = likely wrong correspondences)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)

    fig.suptitle("Feature Association Distance: Edge vs Plane\n"
                 "Large edge–plane gap → edge data association is less reliable",
                 fontsize=11)
    plt.tight_layout()
    _save(fig, out / "association_distances.png")

    # Aggregate histogram over the whole run
    all_edge_mean  = edge_mean[~np.isnan(edge_mean)]
    all_plane_mean = plane_mean[~np.isnan(plane_mean)]
    if all_edge_mean.size and all_plane_mean.size:
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, max(np.percentile(all_edge_mean, 99),
                                  np.percentile(all_plane_mean, 99)) * 1e2 * 1.1, 50)
        ax.hist(all_edge_mean  * 1e2, bins=bins, color="tab:blue",   alpha=0.6,
                label=f"edge  (median {np.median(all_edge_mean)*1e2:.1f} cm)")
        ax.hist(all_plane_mean * 1e2, bins=bins, color="tab:orange", alpha=0.6,
                label=f"plane (median {np.median(all_plane_mean)*1e2:.1f} cm)")
        ax.set_xlabel("Mean association distance per scan (cm)")
        ax.set_ylabel("Scan count")
        ax.set_title("Association Distance Distribution — Edge vs Plane (whole run)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        plt.tight_layout()
        _save(fig, out / "association_distances_hist.png")


def plot_hessian_constraint_quality(stats_data: list, out: Path):
    """Plot Hessian-based constraint quality from ceres::Covariance (J^T J inverse).

    condition_num = sqrt(lambda_min / lambda_max) of the position/orientation
    covariance block.  Close to 1 = well-conditioned (constraints in all directions).
    Close to 0 = degenerate (one or more DoF barely constrained).

    hessian_var_* = diagonal of the 6x6 pose covariance in tangent space.
    Units: metres^2 for position, radians^2 for orientation.
    """
    if not stats_data:
        print("  [skip] hessian constraint quality — no optimisation stats messages")
        return

    t0 = stats_data[0][0]
    t = np.array([s - t0 for s, _ in stats_data])
    msgs = [m for _, m in stats_data]

    def col(f):
        return np.array([getattr(m, f, float("nan")) for m in msgs])

    pos_cond = col("hessian_pos_condition_num")
    ori_cond = col("hessian_ori_condition_num")

    if np.all(np.isnan(pos_cond)):
        print("  [skip] hessian constraint quality — fields not present in bag (rebuild required)")
        return

    # --- condition numbers ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(t, pos_cond, "-", color="tab:blue",   linewidth=0.9, label="position")
    axes[0].plot(t, ori_cond, "-", color="tab:orange", linewidth=0.9, label="orientation")
    axes[0].axhline(0.1, color="tab:red", linestyle="--", linewidth=0.8, label="degeneracy threshold (0.1)")
    axes[0].set_ylabel("√(λ_min / λ_max)")
    axes[0].set_title("Hessian condition number  (1 = fully constrained, 0 = degenerate)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)

    # --- per-DoF variance (std dev for readability) ---
    pos_fields = [("hessian_var_x", "X", "tab:blue"),
                  ("hessian_var_y", "Y", "tab:orange"),
                  ("hessian_var_z", "Z", "tab:green")]
    for f, label, c in pos_fields:
        axes[1].plot(t, np.sqrt(np.abs(col(f))) * 1e3, "-", color=c, linewidth=0.9, label=label)
    axes[1].set_ylabel("Position std dev (mm)")
    axes[1].set_title("Per-axis position uncertainty  (√diagonal of covariance)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)

    fig.suptitle("Hessian-Based Constraint Quality (from Ceres covariance)")
    plt.tight_layout()
    _save(fig, out / "hessian_constraint_quality.png")

    # --- orientation DoF variance ---
    fig, ax = plt.subplots(figsize=(13, 4))
    ori_fields = [("hessian_var_roll",  "Roll",  "tab:red"),
                  ("hessian_var_pitch", "Pitch", "tab:purple"),
                  ("hessian_var_yaw",   "Yaw",   "tab:brown")]
    for f, label, c in ori_fields:
        ax.plot(t, np.sqrt(np.abs(col(f))) * (180.0 / np.pi), "-", color=c,
                linewidth=0.9, label=label)
    ax.set_ylabel("Orientation std dev (°)")
    ax.set_title("Per-axis orientation uncertainty  (√diagonal of covariance)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, out / "hessian_orientation_uncertainty.png")


def plot_imu_vs_lidar(odom_lidar: dict, odom_imu: dict, out: Path):
    """Compare LiDAR odometry against IMU state estimation."""
    if not odom_lidar or not odom_imu:
        print("  [skip] imu vs lidar comparison — missing one source")
        return

    # align time bases
    t_ref = min(odom_lidar["t"][0] if len(odom_lidar["t"]) else 0,
                odom_imu["t"][0] if len(odom_imu["t"]) else 0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, key, ylabel in zip(axes, ["x", "y", "z"], ["X (m)", "Y (m)", "Z (m)"]):
        ax.plot(odom_lidar["t"], odom_lidar[key], "-", color="tab:blue",
                linewidth=0.8, label="laser_odometry (LiDAR)")
        ax.plot(odom_imu["t"], odom_imu[key], "-", color="tab:orange",
                linewidth=0.8, alpha=0.8, label="state_estimation (IMU)")
        ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("LiDAR Odometry vs IMU State Estimation")
    plt.tight_layout()
    _save(fig, out / "imu_vs_lidar.png")


# ---------------------------------------------------------------------------
# Inter-source drift via RPE (evo)
# ---------------------------------------------------------------------------

def _rpy_deg_to_quat_wxyz(roll_deg: np.ndarray, pitch_deg: np.ndarray,
                           yaw_deg: np.ndarray) -> np.ndarray:
    """Convert degree RPY arrays back to quaternions in evo's (w, x, y, z) order."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.column_stack([w, x, y_, z])


def _odom_to_evo_traj(odom: dict) -> "trajectory.PoseTrajectory3D":
    """Convert an extracted odom dict to an evo PoseTrajectory3D."""
    positions = np.column_stack([odom["x"], odom["y"], odom["z"]])
    quats_wxyz = _rpy_deg_to_quat_wxyz(odom["roll"], odom["pitch"], odom["yaw"])
    # evo timestamps must be absolute (not relative) for association
    # odom["t"] is relative; add a fixed offset so association works correctly
    return trajectory.PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quats_wxyz,
        timestamps=odom["t"],
    )


def plot_inter_source_drift(odom_ref: dict, odom_est: dict,
                             ref_label: str, est_label: str,
                             out: Path,
                             delta_m: float = 1.0) -> None:
    """
    Inter-source drift via Relative Pose Error (RPE).

    Pairs odom_ref and odom_est by timestamp (same moment, two sensors) and
    computes the relative translation error between them over every `delta_m`-
    metre segment.  This measures how differently the two sources describe the
    same motion — i.e. how fast they drift apart — not whether the robot
    revisited a physical location.

    Parameters
    ----------
    odom_ref  : extract_odom() dict — treated as the more trusted source
    odom_est  : extract_odom() dict — source being evaluated (e.g. IMU odom)
    ref_label : legend label for odom_ref
    est_label : legend label for odom_est
    out       : output directory Path
    delta_m   : segment length in metres for RPE evaluation
    """
    if not HAS_EVO:
        print("  [skip] inter-source drift — evo not installed")
        return
    if not odom_ref or not odom_est:
        print(f"  [skip] inter-source drift — missing {ref_label} or {est_label}")
        return

    traj_ref = _odom_to_evo_traj(odom_ref)
    traj_est = _odom_to_evo_traj(odom_est)

    # Associate (temporally sync) the two trajectories
    try:
        traj_ref_s, traj_est_s = sync.associate_trajectories(
            traj_ref, traj_est, max_diff=0.1
        )
    except Exception as e:
        print(f"  [skip] inter-source drift — trajectory association failed: {e}")
        return

    if len(traj_ref_s.timestamps) < 4:
        print("  [skip] inter-source drift — too few matched poses after sync")
        return

    # RPE: translation error per metre-spaced segment
    rpe_metric = metrics.RPE(
        pose_relation=PoseRelation.translation_part,
        delta=delta_m,
        delta_unit=Unit.meters,
        all_pairs=False,
    )
    try:
        rpe_metric.process_data((traj_ref_s, traj_est_s))
    except Exception as e:
        print(f"  [skip] inter-source drift — RPE computation failed: {e}")
        return

    stats = rpe_metric.get_all_statistics()
    errors = np.array(rpe_metric.error)          # per-segment translation error (m)
    seg_t  = traj_ref_s.timestamps[:len(errors)] # matching timestamps

    # --- print summary ---
    print(f"\n  Inter-source drift RPE ({ref_label} vs {est_label}, δ={delta_m} m):")
    for k, v in stats.items():
        print(f"    {k:>8s}: {v:.4f} m")

    # --- save stats to text ---
    stats_file = out / "rpe_inter_source_drift.txt"
    with open(stats_file, "w") as f:
        f.write(f"Inter-source drift RPE — {ref_label} vs {est_label}\n")
        f.write(f"delta = {delta_m} m, pose_relation = translation_part\n")
        f.write("Measures how differently two sources describe the same motion\n"
                "at the same timestamp — not revisited physical locations.\n\n")
        for k, v in stats.items():
            f.write(f"{k}: {v:.6f} m\n")
    print(f"  saved: {stats_file}")

    # --- per-segment error plot ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    # Top: error over time
    ax = axes[0]
    ax.plot(seg_t, errors, "-", color="tab:red", linewidth=0.8, label="RPE per segment")
    ax.axhline(stats["mean"], color="black", linewidth=0.8, linestyle="--",
               label=f"mean {stats['mean']:.3f} m")
    ax.fill_between(seg_t,
                    stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                    alpha=0.15, color="tab:red", label=f"±1σ ({stats['std']:.3f} m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Translation error (m)")
    ax.set_title(f"Inter-source drift — {ref_label} vs {est_label} (δ={delta_m} m segments)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Bottom: error histogram
    ax = axes[1]
    ax.hist(errors, bins=40, color="tab:blue", edgecolor="white", linewidth=0.3)
    ax.axvline(stats["mean"],   color="black", linewidth=1.0, linestyle="--",
               label=f"mean {stats['mean']:.3f} m")
    ax.axvline(stats["median"], color="tab:orange", linewidth=1.0, linestyle="-.",
               label=f"median {stats['median']:.3f} m")
    ax.axvline(stats["rmse"],   color="tab:red", linewidth=1.0, linestyle=":",
               label=f"RMSE {stats['rmse']:.3f} m")
    ax.set_xlabel("Translation error (m)")
    ax.set_ylabel("Segment count")
    ax.set_title("Error distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    fig.suptitle(f"Inter-source Drift (RPE) — {ref_label} vs {est_label}", fontsize=13)
    plt.tight_layout()
    _save(fig, out / "rpe_inter_source_drift.png")

    # --- save raw per-segment errors as CSV ---
    csv_path = out / "rpe_inter_source_drift.csv"
    np.savetxt(
        csv_path,
        np.column_stack([seg_t, errors]),
        delimiter=",",
        header="t,rpe_translation_m",
        comments="",
    )
    print(f"  saved: {csv_path}")


# ---------------------------------------------------------------------------
# Map consistency — submap ICP + MME  (optional, requires open3d)
# ---------------------------------------------------------------------------

def _pc2_to_xyz(msg) -> np.ndarray:
    """Extract (N,3) float32 XYZ array from a sensor_msgs/PointCloud2 message."""
    from sensor_msgs_py import point_cloud2 as pc2_reader
    # read_points returns a structured array with named fields — extract each
    # column individually then stack, rather than casting the structured dtype.
    pts = pc2_reader.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pts = np.asarray(pts)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)



def _find_all_loop_return_times(odom: dict, x_tol: float, min_loop_t: float) -> list[float]:
    """
    Automatically detect all loop return times from the odometry trajectory.

    Strategy: compute the 2D Euclidean distance from the starting position
    over time, then find valleys (local minima) in that signal.  Each valley
    where the robot is within `max_return_dist` of the start corresponds to
    one loop return.  Using distance rather than a single-axis threshold makes
    this robust across datasets with different orientations and drift levels.

    Parameters
    ----------
    odom         : extract_odom() dict
    x_tol        : kept for API compatibility; used as max_return_dist — the
                   maximum 2D distance from start that counts as a return.
                   If set to 0 (default sentinel), it is estimated automatically
                   as 15% of the maximum distance reached in the trajectory.
    min_loop_t   : minimum time in seconds between consecutive loop returns.
                   Set to ~80% of the expected single-loop duration.
                   If uncertain, set lower rather than higher — false positives
                   are filtered by the distance threshold.
    """
    from scipy.signal import find_peaks

    if not odom:
        return []

    t  = odom["t"]
    x0, y0 = odom["x"][0], odom["y"][0]

    # 2D distance from start at every timestep
    dist = np.sqrt((odom["x"] - x0) ** 2 + (odom["y"] - y0) ** 2)

    # Auto-set max_return_dist if caller passed 0 or left default
    max_return_dist = x_tol if x_tol > 0 else float(np.max(dist) * 0.15)

    # Convert min_loop_t to a sample count for find_peaks
    dt_median = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    min_samples = max(1, int(min_loop_t / dt_median))

    # Find valleys in the distance signal: invert so valleys become peaks
    # prominence filters out shallow dips; distance enforces min separation
    peaks, props = find_peaks(
        -dist,
        distance=min_samples,
        prominence=max_return_dist * 0.3,   # valley must drop by at least 30% of return radius
    )

    # Keep only valleys where the robot actually got close to the start
    loop_times = []
    for idx in peaks:
        if dist[idx] <= max_return_dist and t[idx] > 0:
            loop_times.append(float(t[idx]))

    print(f"  Loop detection: max_return_dist={max_return_dist:.2f} m, "
          f"min_loop_t={min_loop_t:.0f} s → {len(loop_times)} loop return(s) found")
    if loop_times:
        print(f"    Return times: {[f'{lt:.1f}s' for lt in loop_times]}")

    return loop_times


def _compute_mme(points: np.ndarray, k: int = 10, max_pts: int = 30_000) -> float:
    """
    Map Mean Entropy (MME) — mean log-determinant of the local covariance
    estimated from the k nearest neighbours of each point.
    Lower = sharper / more consistent map.

    Points are randomly subsampled to `max_pts` before computing to avoid
    killing the process on large maps.
    """
    from scipy.spatial import KDTree
    if len(points) < k + 1:
        return float("nan")
    if len(points) > max_pts:
        idx = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        print(f"    MME: subsampled to {max_pts:,} pts for tractable computation")
    tree = KDTree(points)
    _, indices = tree.query(points, k=k + 1, workers=-1)  # use all CPU cores
    entropies = []
    for idx in indices:
        neighbours = points[idx[1:]]   # exclude self
        cov = np.cov(neighbours.T)
        sign, logdet = np.linalg.slogdet(cov)
        if sign > 0:
            entropies.append(logdet)
    return float(np.mean(entropies)) if entropies else float("nan")


def save_global_map_pcd(msg, out: Path) -> None:
    """
    Save the global accumulated map as a PCD file.

    Expects the last laser_cloud_map message (a single PointCloud2), which
    contains the most complete version of the map.  Requires open3d.
    """
    if not HAS_O3D:
        print("  [skip] save map — open3d not installed (pip install open3d)")
        return
    if msg is None:
        print("  [skip] save map — no laser_cloud_map messages in bag")
        return

    pts = _pc2_to_xyz(msg)
    if pts.size == 0:
        print("  [skip] save map — last laser_cloud_map message is empty")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    pcd_path = out / "map.pcd"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"  saved: {pcd_path}  ({len(pcd.points):,} points)")


def _rpy_to_rot(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,          1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0         ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0,          0         ],
                   [0, np.cos(r), -np.sin(r) ],
                   [0, np.sin(r),  np.cos(r) ]])
    return Rz @ Ry @ Rx


def _icp_align_submaps(pcd_src, pcd_ref, odom_laser, t_src_rel, max_dist=0.5):
    """
    Coarse-initialise from odometry drift at t_src_rel, then refine with ICP.
    Returns (icp_result, T_init).
    """
    odom_t   = np.asarray(odom_laser["t"])
    idx      = int(np.argmin(np.abs(odom_t - t_src_rel)))
    drift_x  = float(odom_laser["x"][idx] - odom_laser["x"][0])
    drift_y  = float(odom_laser["y"][idx] - odom_laser["y"][0])
    drift_z  = float(odom_laser["z"][idx] - odom_laser["z"][0])

    R0      = _rpy_to_rot(odom_laser["roll"][0],      odom_laser["pitch"][0],      odom_laser["yaw"][0])
    R_idx   = _rpy_to_rot(odom_laser["roll"][idx],    odom_laser["pitch"][idx],    odom_laser["yaw"][idx])
    R_drift = R0.T @ R_idx

    T_init = np.eye(4)
    T_init[:3, :3] = R_drift.T
    T_init[:3,  3] = [-drift_x, -drift_y, -drift_z]

    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_ref,
        max_correspondence_distance=max_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return icp_result, T_init


def plot_loop_drift(odom_laser: dict, out: Path,
                    x_tol: float = 1.0, min_loop_t: float = 30.0) -> None:
    """
    Loop return drift — no ground truth required.

    Each time the robot physically returns to the starting position, the
    odometry's reported position is compared against the starting pose.
    The difference is the accumulated drift for that loop.

    If error compounds, the drift at loop 2 will be larger than loop 1,
    and loop 3 larger still.  If the algorithm handles degeneracy well,
    drift should stay roughly constant across loops.

    Produces:
      loop_drift.png  — two panels:
        Left  : XY trajectory with loop return points marked and annotated
                with their drift magnitude.
        Right : Bar chart of translational drift per loop return, plus a
                breakdown into X / Y / Z components so you can see which
                axis is accumulating the most error.
      loop_drift.csv  — one row per loop return with full drift details.
    """
    if not odom_laser:
        print("  [skip] loop drift — no laser odometry")
        return

    loop_times = _find_all_loop_return_times(odom_laser, x_tol=x_tol, min_loop_t=min_loop_t)
    if not loop_times:
        print(f"  [skip] loop drift — no loop returns detected "
              f"(x_tol={x_tol} m, min_loop_t={min_loop_t} s)")
        return

    odom_t = np.asarray(odom_laser["t"])
    x0, y0, z0 = odom_laser["x"][0], odom_laser["y"][0], odom_laser["z"][0]

    rows = []
    for loop_idx, lt in enumerate(loop_times):
        idx = int(np.argmin(np.abs(odom_t - lt)))
        dx  = float(odom_laser["x"][idx] - x0)
        dy  = float(odom_laser["y"][idx] - y0)
        dz  = float(odom_laser["z"][idx] - z0)
        mag = float(np.sqrt(dx**2 + dy**2 + dz**2))
        rows.append(dict(loop=loop_idx + 1, t=lt, dx=dx, dy=dy, dz=dz, mag=mag,
                         x_ret=float(odom_laser["x"][idx]),
                         y_ret=float(odom_laser["y"][idx])))
        print(f"  Loop {loop_idx+1} return at t={lt:.1f}s: "
              f"drift=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m  |mag|={mag:.3f} m")

    # ---- CSV ----
    csv_path = out / "loop_drift.csv"
    with open(csv_path, "w") as f:
        f.write("loop,t,dx,dy,dz,magnitude\n")
        for r in rows:
            f.write(f"{r['loop']},{r['t']:.3f},{r['dx']:.6f},{r['dy']:.6f},"
                    f"{r['dz']:.6f},{r['mag']:.6f}\n")
    print(f"  saved: {csv_path}")

    # ---- figure ----
    LOOP_COLORS = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: XY trajectory with loop return markers
    ax = axes[0]
    ax.plot(odom_laser["x"], odom_laser["y"], "-", color="tab:blue",
            linewidth=0.7, alpha=0.6, label="trajectory")
    ax.plot(x0, y0, "k*", markersize=10, zorder=5, label="start / true return")

    for r, c in zip(rows, LOOP_COLORS):
        ax.plot(r["x_ret"], r["y_ret"], "o", color=c, markersize=8, zorder=5,
                label=f"Loop {r['loop']} return (drift {r['mag']:.3f} m)")
        # Arrow from true start to estimated return position
        ax.annotate(
            "",
            xy=(r["x_ret"], r["y_ret"]),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.5),
        )
        ax.text(r["x_ret"] + 0.1, r["y_ret"] + 0.1,
                f"L{r['loop']}: {r['mag']:.2f} m", fontsize=8, color=c)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Loop Return Drift\n(arrow = estimated position vs true start)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # Right: bar chart — total drift magnitude + XYZ breakdown
    ax2 = axes[1]
    loop_labels = [f"Loop {r['loop']}\n(t={r['t']:.0f}s)" for r in rows]
    x_pos = np.arange(len(rows))
    width = 0.2

    mags = [r["mag"] for r in rows]
    dxs  = [abs(r["dx"]) for r in rows]
    dys  = [abs(r["dy"]) for r in rows]
    dzs  = [abs(r["dz"]) for r in rows]

    b0 = ax2.bar(x_pos - 1.5*width, mags, width, label="|drift|",  color="tab:blue")
    b1 = ax2.bar(x_pos - 0.5*width, dxs,  width, label="|dx|",     color="tab:red",    alpha=0.8)
    b2 = ax2.bar(x_pos + 0.5*width, dys,  width, label="|dy|",     color="tab:orange", alpha=0.8)
    b3 = ax2.bar(x_pos + 1.5*width, dzs,  width, label="|dz|",     color="tab:green",  alpha=0.8)

    for bars in [b0, b1, b2, b3]:
        ax2.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax2.set_xticks(x_pos); ax2.set_xticklabels(loop_labels, fontsize=9)
    ax2.set_ylabel("Drift (m)")
    ax2.set_title("Accumulated Drift per Loop Return\n"
                  "(compounding = each bar taller than the last)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Loop Return Drift — Estimated Position vs True Start\n"
        "(no ground truth needed: robot physically returned to start)",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, out / "loop_drift.png")


def plot_map_consistency(submaps_pts: list[np.ndarray], loop_labels: list[str],
                         loop_times: list[float],
                         odom_laser: dict, out: Path,
                         compute_mme: bool = False,
                         mme_max_pts: int = 30_000,
                         icp_max_dist: float = 0.5) -> None:
    """
    Multi-loop map consistency metric (requires open3d).

    Takes pre-built submaps (one np.ndarray per loop pass, already voxel-
    downsampled by SubmapBuilderSubscriber) and runs pairwise ICP between
    every pair.  Loop detection and submap building happen in main() so that
    scans are never all held in memory simultaneously.

    For each pair (i → j):
      - Coarse-initialise the alignment from the accumulated odometry drift
        at the loop boundary.
      - Refine with point-to-point ICP.
      - Report fitness, inlier RMSE, translation & rotation drift, and
        post-alignment NN RMSE.

    Also computes MME on the merged map if --mme is set.
    """
    if not HAS_O3D:
        print("  [skip] map consistency — open3d not installed (pip install open3d)")
        return
    if not submaps_pts:
        print("  [skip] map consistency — no submaps provided")
        return
    if not odom_laser:
        print("  [skip] map consistency — no laser_odometry data")
        return

    n_submaps = len(submaps_pts)

    # ---- wrap raw arrays into open3d point clouds ----
    def _to_o3d(pts: np.ndarray):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return pc

    submaps_pcd = []
    for i, pts in enumerate(submaps_pts):
        pcd = _to_o3d(pts) if len(pts) >= 100 else None
        submaps_pcd.append(pcd)
        if pcd:
            print(f"    Submap {loop_labels[i]}: {len(pts):,} pts")
        else:
            print(f"    Submap {loop_labels[i]}: {len(pts):,} pts — TOO SPARSE, skipping")

    # ---- pairwise ICP ----
    # All pairs (i, j) where j > i — e.g. (A,B), (B,C), (A,C) for 3 loops.
    pairs = [(i, j) for i in range(n_submaps) for j in range(i + 1, n_submaps)]

    stats_file = out / "map_consistency.txt"
    all_pair_results = []

    with open(stats_file, "w") as f:
        f.write(f"Map consistency — {n_submaps} submaps, pairwise ICP\n")
        f.write(f"Loop return times: {[f'{lt:.1f}s' for lt in loop_times]}\n\n")

        for (i, j) in pairs:
            lbl_i, lbl_j = loop_labels[i], loop_labels[j]
            pcd_ref = submaps_pcd[i]
            pcd_src = submaps_pcd[j]

            if pcd_ref is None or pcd_src is None:
                print(f"  [skip] pair ({lbl_i},{lbl_j}) — submap too sparse")
                f.write(f"Pair ({lbl_i},{lbl_j}): skipped — submap too sparse\n\n")
                all_pair_results.append(None)
                continue

            # The loop boundary for submap j is at loop_times[j-1]
            t_boundary_rel = loop_times[j - 1]
            print(f"\n  Pair ({lbl_i} → {lbl_j}): ICP alignment …")
            icp_result, _ = _icp_align_submaps(
                pcd_src, pcd_ref, odom_laser, t_boundary_rel, icp_max_dist
            )
            T = icp_result.transformation
            t_drift     = float(np.linalg.norm(T[:3, 3]))
            angle_drift = float(np.degrees(np.arccos(
                np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            )))

            pcd_src_aligned = o3d.geometry.PointCloud(pcd_src).transform(T)
            dists   = np.asarray(pcd_src_aligned.compute_point_cloud_distance(pcd_ref))
            rmse    = float(np.sqrt(np.mean(dists ** 2)))
            mean_nn = float(np.mean(dists))
            p95     = float(np.percentile(dists, 95))

            print(f"    ICP fitness      : {icp_result.fitness:.4f}")
            print(f"    ICP inlier RMSE  : {icp_result.inlier_rmse:.4f} m")
            print(f"    Drift (trans)    : {t_drift:.4f} m")
            print(f"    Drift (rot)      : {angle_drift:.3f} deg")
            print(f"    Post-align RMSE  : {rmse:.4f} m")

            f.write(f"Pair ({lbl_i},{lbl_j}):\n")
            f.write(f"  Submap {lbl_i}: {len(submaps_pcd[i].points):,} pts (downsampled)\n")
            f.write(f"  Submap {lbl_j}: {len(submaps_pcd[j].points):,} pts (downsampled)\n")
            f.write(f"  ICP fitness          : {icp_result.fitness:.6f}\n")
            f.write(f"  ICP inlier RMSE      : {icp_result.inlier_rmse:.6f} m\n")
            f.write(f"  Drift translation    : {t_drift:.6f} m\n")
            f.write(f"  Drift rotation       : {angle_drift:.4f} deg\n")
            f.write(f"  Post-alignment RMSE  : {rmse:.6f} m\n")
            f.write(f"  Post-alignment mean NN: {mean_nn:.6f} m\n")
            f.write(f"  Post-alignment P95 NN: {p95:.6f} m\n\n")

            all_pair_results.append({
                "lbl_i": lbl_i, "lbl_j": lbl_j,
                "pcd_ref": pcd_ref, "pcd_src_aligned": pcd_src_aligned,
                "pts_ref": submaps_pts[i], "pts_src": submaps_pts[j],
                "dists": dists,
                "icp_fitness": icp_result.fitness,
                "icp_rmse": icp_result.inlier_rmse,
                "t_drift": t_drift, "angle_drift": angle_drift,
                "rmse": rmse, "mean_nn": mean_nn, "p95": p95,
            })

        # ---- MME on full merged map ----
        if compute_mme:
            all_pts = np.vstack([p for p in submaps_pts if len(p) >= 100])
            print(f"\n  Computing MME on {len(all_pts):,} merged points …")
            mme_val = _compute_mme(all_pts, max_pts=mme_max_pts)
            print(f"    MME: {mme_val:.4f} (lower = sharper map)")
            f.write(f"MME (full merged map): {mme_val:.6f}\n")

    print(f"  saved: {stats_file}")

    # ---- figures: one per pair ----
    SUBMAP_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for result in all_pair_results:
        if result is None:
            continue
        lbl_i, lbl_j = result["lbl_i"], result["lbl_j"]
        pts_ref = result["pts_ref"]
        pts_src = np.asarray(result["pcd_src_aligned"].points)
        dists   = result["dists"]
        c_ref   = SUBMAP_COLORS[ord(lbl_i) - ord("A")]
        c_src   = SUBMAP_COLORS[ord(lbl_j) - ord("A")]

        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(2, 2, figure=fig)

        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(pts_ref[::10, 0], pts_ref[::10, 1], s=0.3, c=c_ref, alpha=0.4)
        ax.set_title(f"Submap {lbl_i} — pass {ord(lbl_i)-ord('A')+1} (XY)")
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(pts_src[::10, 0], pts_src[::10, 1], s=0.3, c=c_src, alpha=0.4)
        ax.set_title(f"Submap {lbl_j} — ICP-aligned  "
                     f"(drift {result['t_drift']:.3f} m / {result['angle_drift']:.2f}°)")
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

        ax = fig.add_subplot(gs[1, 0])
        ax.scatter(pts_ref[::10, 0], pts_ref[::10, 1], s=0.3, c=c_ref,  alpha=0.35, label=lbl_i)
        ax.scatter(pts_src[::10, 0], pts_src[::10, 1], s=0.3, c=c_src,  alpha=0.35, label=f"{lbl_j} (aligned)")
        ax.set_title("Overlay after ICP alignment")
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.legend(fontsize=8, markerscale=5)

        ax = fig.add_subplot(gs[1, 1])
        sc = ax.scatter(pts_src[::10, 0], pts_src[::10, 1],
                        s=0.3, c=dists[::10], cmap="hot_r",
                        vmin=0, vmax=np.percentile(dists, 99), alpha=0.6)
        plt.colorbar(sc, ax=ax, label="NN dist (m)")
        ax.set_title(f"{lbl_j} coloured by NN dist to {lbl_i}  (RMSE={result['rmse']:.3f} m)")
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

        fig.suptitle(
            f"Map Consistency — Submaps {lbl_i} vs {lbl_j}  |  "
            f"ICP drift: {result['t_drift']:.3f} m / {result['angle_drift']:.2f}°  |  "
            f"Post-align NN RMSE: {result['rmse']:.3f} m",
            fontsize=11,
        )
        plt.tight_layout()
        _save(fig, out / f"map_consistency_{lbl_i}{lbl_j}.png")

    # ---- drift summary bar chart (if more than one pair) ----
    valid_results = [r for r in all_pair_results if r is not None]
    if len(valid_results) > 1:
        pair_labels  = [f"{r['lbl_i']}↔{r['lbl_j']}" for r in valid_results]
        drift_trans  = [r["t_drift"]  for r in valid_results]
        drift_rot    = [r["angle_drift"] for r in valid_results]
        post_rmse    = [r["rmse"]     for r in valid_results]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, vals, ylabel, title in zip(
            axes,
            [drift_trans, drift_rot, post_rmse],
            ["Translation drift (m)", "Rotation drift (°)", "Post-align NN RMSE (m)"],
            ["ICP Translation Drift per Pair",
             "ICP Rotation Drift per Pair",
             "Post-alignment Surface RMSE per Pair"],
        ):
            bars = ax.bar(pair_labels, vals, color=["tab:blue", "tab:orange", "tab:green"][:len(vals)])
            ax.bar_label(bars, fmt="%.3f", fontsize=9)
            ax.set_ylabel(ylabel); ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Pairwise Drift Summary — Does Error Compound Across Loops?", fontsize=12)
        plt.tight_layout()
        _save(fig, out / "map_consistency_summary.png")


# ---------------------------------------------------------------------------
# Combined uncertainty + trajectory correlation
# ---------------------------------------------------------------------------

def plot_uncertainty_with_trajectory(stats_data: list, odom_laser: dict, out: Path) -> None:
    """
    Three-panel plot correlating pose uncertainty with position over time.

    Top    : X / Y / Z uncertainty on the same axes — so you can compare
             which axis is most uncertain at any moment.
    Middle : X / Y / Z position vs time from laser odometry.
    Bottom : XY trajectory coloured by total uncertainty magnitude
             (sqrt(unc_x^2 + unc_y^2 + unc_z^2)) so you can see *where*
             in the map the estimator was least confident.
    """
    if not stats_data:
        print("  [skip] uncertainty+trajectory — no stats messages")
        return
    if not odom_laser:
        print("  [skip] uncertainty+trajectory — no laser odometry")
        return

    t0 = stats_data[0][0]
    t_unc = np.array([s - t0 for s, _ in stats_data])
    msgs  = [m for _, m in stats_data]

    unc_x = np.array([m.uncertainty_x     for m in msgs])
    unc_y = np.array([m.uncertainty_y     for m in msgs])
    unc_z = np.array([m.uncertainty_z     for m in msgs])
    unc_mag = np.sqrt(unc_x**2 + unc_y**2 + unc_z**2)

    # Interpolate uncertainty magnitude onto odometry timestamps for spatial plot
    odom_t = odom_laser["t"]
    unc_mag_interp = np.interp(odom_t, t_unc, unc_mag,
                               left=float("nan"), right=float("nan"))

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2, 2, 3], hspace=0.35)

    # ---- top: all three uncertainties on one axes ----
    ax = fig.add_subplot(gs[0])
    ax.plot(t_unc, unc_x, "-", color="tab:blue",   linewidth=0.8, label="Unc X")
    ax.plot(t_unc, unc_y, "-", color="tab:orange",  linewidth=0.8, label="Unc Y")
    ax.plot(t_unc, unc_z, "-", color="tab:green",   linewidth=0.8, label="Unc Z")
    ax.plot(t_unc, unc_mag, "-", color="black", linewidth=0.6,
            alpha=0.5, linestyle="--", label="|unc| magnitude")
    ax.set_ylabel("Uncertainty")
    ax.set_title("Position Uncertainty X / Y / Z over Time")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.4)

    # ---- middle: position vs time ----
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.plot(odom_t, odom_laser["x"], "-", color="tab:blue",   linewidth=0.8, label="X")
    ax2.plot(odom_t, odom_laser["y"], "-", color="tab:orange",  linewidth=0.8, label="Y")
    ax2.plot(odom_t, odom_laser["z"], "-", color="tab:green",   linewidth=0.8, label="Z")
    ax2.set_ylabel("Position (m)")
    ax2.set_title("Laser Odometry Position vs Time")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.4)
    ax2.set_xlabel("Time (s)")

    # ---- bottom: XY trajectory coloured by uncertainty magnitude ----
    ax3 = fig.add_subplot(gs[2])
    valid = ~np.isnan(unc_mag_interp)
    sc = ax3.scatter(
        odom_laser["x"][valid], odom_laser["y"][valid],
        c=unc_mag_interp[valid], cmap="YlOrRd",
        s=2, linewidths=0,
        vmin=0, vmax=np.nanpercentile(unc_mag_interp, 95),
    )
    plt.colorbar(sc, ax=ax3, label="|uncertainty| magnitude")
    ax3.plot(odom_laser["x"][0], odom_laser["y"][0], "go", markersize=6, label="start")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)")
    ax3.set_title("Trajectory coloured by Uncertainty Magnitude  (bright = less confident)")
    ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3); ax3.legend(fontsize=8)

    fig.suptitle("Pose Uncertainty Correlated with Trajectory", fontsize=13)
    _save(fig, out / "uncertainty_trajectory.png")


# ---------------------------------------------------------------------------
# Prediction source heatmap over spatial trajectory
# ---------------------------------------------------------------------------

def plot_prediction_source_heatmap(stats_data: list, odom_laser: dict, out: Path) -> None:
    """
    XY trajectory coloured by which odometry source the estimator was using
    (LiDAR-inertial vs VIO vs IMU-only), so you can see spatially where
    each source dominated and correlate that with drift or uncertainty.

    Prediction source codes (from SuperOdometry):
        0 = IMU_ORIENT   — orientation-only IMU propagation
        1 = LIO_ODOM     — LiDAR-inertial odometry
        2 = VIO_ODOM     — visual-inertial odometry
        3 = NEURAL_IMU   — neural IMU prediction
        4 = CONST_VEL    — constant velocity fallback
    """
    if not stats_data:
        print("  [skip] prediction source heatmap — no stats messages")
        return
    if not odom_laser:
        print("  [skip] prediction source heatmap — no laser odometry")
        return

    PRED_LABELS = {0: "IMU_ORIENT", 1: "LIO_ODOM", 2: "VIO_ODOM",
                   3: "NEURAL_IMU", 4: "CONST_VEL"}
    PRED_COLORS = {0: "tab:purple", 1: "tab:blue", 2: "tab:orange",
                   3: "tab:green",  4: "tab:red"}

    t0    = stats_data[0][0]
    t_src = np.array([s - t0 for s, _ in stats_data])
    src   = np.array([m.prediction_source for _, m in stats_data], dtype=float)

    # Interpolate source onto odometry timestamps (nearest-neighbour)
    src_interp = np.interp(odom_laser["t"], t_src, src)
    src_int    = np.round(src_interp).astype(int)

    present_sources = sorted(set(src_int.tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- left: XY trajectory coloured by source ----
    ax = axes[0]
    for code in present_sources:
        mask = src_int == code
        label = PRED_LABELS.get(code, str(code))
        color = PRED_COLORS.get(code, "gray")
        ax.scatter(
            odom_laser["x"][mask], odom_laser["y"][mask],
            s=2, color=color, label=label, linewidths=0,
        )
    ax.plot(odom_laser["x"][0], odom_laser["y"][0], "k*", markersize=8, label="start")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory — coloured by Active Prediction Source")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, markerscale=4, loc="best")

    # ---- right: fraction of time each source was active (bar chart) ----
    ax2 = axes[1]
    total = len(src_int)
    fracs = {PRED_LABELS.get(c, str(c)): np.sum(src_int == c) / total * 100
             for c in present_sources}
    bar_labels = list(fracs.keys())
    bar_vals   = list(fracs.values())
    bar_colors = [PRED_COLORS.get(c, "gray") for c in present_sources]
    bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="white")
    ax2.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax2.set_ylabel("% of trajectory")
    ax2.set_title("Prediction Source — Time Fraction")
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Active Prediction Source: LiDAR vs IMU vs VIO", fontsize=13)
    plt.tight_layout()
    _save(fig, out / "prediction_source_heatmap.png")


# ---------------------------------------------------------------------------
# Feature geometry metrics
# ---------------------------------------------------------------------------

def _pc2_msg_to_xyz(pc2_msg) -> np.ndarray:
    """Extract (N,3) float32 array from a sensor_msgs/PointCloud2 message."""
    from sensor_msgs_py import point_cloud2 as pc2_reader
    pts_struct = pc2_reader.read_points(pc2_msg, field_names=("x", "y", "z"), skip_nans=True)
    if pts_struct.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.column_stack([
        np.asarray(pts_struct["x"], dtype=np.float32),
        np.asarray(pts_struct["y"], dtype=np.float32),
        np.asarray(pts_struct["z"], dtype=np.float32),
    ])


def _geometry_stats(pts: np.ndarray) -> tuple[float, float]:
    """Return (eig_ratio, eig_min) for a point cloud array."""
    if len(pts) < 4:
        return float("nan"), float("nan")
    cov = np.cov(pts.T)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    eig_max = eigvals[-1]
    ratio = float(eigvals[0] / eig_max) if eig_max > 1e-9 else 0.0
    return ratio, float(eigvals[0])


def extract_feature_geometry_from_feature_info(feature_info_data: list) -> dict:
    """
    Extract per-frame geometry stats for each point cloud embedded in
    /feature_info (LaserFeature messages).

    The individual /bob_points, /edge_points, /planner_points topics are
    never actually published by SuperOdometry — the clouds are packed into
    the LaserFeature message and published via /feature_info only.

    Also note: in the OS1 pipeline only cloud_surface (planner/uniform
    downsample) is populated.  cloud_corner and cloud_realsense are always
    empty because edge/BOB extraction is not implemented for this sensor.

    Returns a dict keyed by cloud name, each value is:
      t         : timestamps (relative, seconds)
      count     : points per frame
      eig_ratio : min/max eigenvalue ratio (0=degenerate, 1=rich)
      eig_min   : smallest eigenvalue
    """
    if not feature_info_data:
        return {}

    t0 = feature_info_data[0][0]
    clouds = {
        "nodistortion": ([], [], [], []),
        "corner":       ([], [], [], []),
        "surface":      ([], [], [], []),
        "realsense":    ([], [], [], []),
    }
    cloud_fields = {
        "nodistortion": "cloud_nodistortion",
        "corner":       "cloud_corner",
        "surface":      "cloud_surface",
        "realsense":    "cloud_realsense",
    }

    # combined = merge of all feature clouds (excludes nodistortion — that's
    # the raw scan, not a feature cloud, so it would dominate the combined stats)
    combined = ([], [], [], [])

    for stamp, msg in feature_info_data:
        t_rel = stamp - t0
        frame_feature_pts = []   # collect feature clouds for combined metric

        for key, field in cloud_fields.items():
            t_list, cnt_list, ratio_list, emin_list = clouds[key]
            pc2_msg = getattr(msg, field, None)
            pts = _pc2_msg_to_xyz(pc2_msg) if pc2_msg is not None else np.empty((0, 3))
            ratio, emin = _geometry_stats(pts)
            t_list.append(t_rel)
            cnt_list.append(len(pts))
            ratio_list.append(ratio)
            emin_list.append(emin)

            # accumulate feature pts for combined (skip nodistortion = raw scan)
            if key != "nodistortion" and len(pts) > 0:
                frame_feature_pts.append(pts)

        # combined degeneracy across all populated feature types this frame
        t_c, cnt_c, ratio_c, emin_c = combined
        if frame_feature_pts:
            merged = np.vstack(frame_feature_pts)
            c_ratio, c_emin = _geometry_stats(merged)
            t_c.append(t_rel)
            cnt_c.append(len(merged))
            ratio_c.append(c_ratio)
            emin_c.append(c_emin)
        else:
            t_c.append(t_rel)
            cnt_c.append(0)
            ratio_c.append(float("nan"))
            emin_c.append(float("nan"))

    result = {}
    for key, (t_list, cnt_list, ratio_list, emin_list) in clouds.items():
        arr_cnt = np.array(cnt_list, dtype=float)
        if arr_cnt.sum() == 0:
            continue   # skip clouds that are always empty
        result[key] = dict(
            t=np.array(t_list),
            count=arr_cnt,
            eig_ratio=np.array(ratio_list),
            eig_min=np.array(emin_list),
        )

    # add combined only if at least one feature cloud was populated
    t_c, cnt_c, ratio_c, emin_c = combined
    arr_cnt_c = np.array(cnt_c, dtype=float)
    if arr_cnt_c.sum() > 0:
        result["combined"] = dict(
            t=np.array(t_c),
            count=arr_cnt_c,
            eig_ratio=np.array(ratio_c),
            eig_min=np.array(emin_c),
        )

    return result


def plot_feature_geometry(feature_stats: dict, out: Path) -> None:
    """
    Plots per-frame point counts and spatial degeneracy scores for each
    non-empty cloud extracted from /feature_info.

      Top   : point count per frame — how many points per cloud type
      Bottom: eigenvalue ratio (min/max) of the per-frame covariance.
              Near 0 → degenerate (flat corridor). Near 1 → feature-rich.
    """
    COLORS = {
        "nodistortion": "tab:blue",
        "corner":       "tab:orange",
        "surface":      "tab:green",
        "realsense":    "tab:red",
        "combined":     "black",
    }
    LABELS = {
        "nodistortion": "Undistorted scan",
        "corner":       "Corner / edge",
        "surface":      "Surface / planar",
        "realsense":    "Depth / BOB",
        "combined":     "Combined features",
    }
    sources = [
        (LABELS.get(k, k), v, COLORS.get(k, "gray"))
        for k, v in feature_stats.items()
    ]

    # ---- feature counts (all three on one plot) ----
    fig, ax = plt.subplots(figsize=(13, 4))
    for label, stats, c in sources:
        if stats:
            ax.plot(stats["t"], stats["count"], "-", color=c, linewidth=0.8, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Points per frame")
    ax.set_title("Feature Point Counts per Frame")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, out / "feature_counts.png")

    # ---- spatial degeneracy score ----
    # One subplot per feature type so the scales don't interfere
    valid = [(l, s, c) for l, s, c in sources if s]
    if not valid:
        print("  [skip] feature geometry — no point cloud data")
        return

    fig, axes = plt.subplots(len(valid), 1, figsize=(13, 4 * len(valid)), sharex=True)
    if len(valid) == 1:
        axes = [axes]

    for ax, (label, stats, c) in zip(axes, valid):
        ratio = stats["eig_ratio"]
        lw = 1.4 if label == "Combined features" else 0.8
        ax.plot(stats["t"], ratio, "-", color=c, linewidth=lw, label=label)
        ax.axhline(np.nanmean(ratio), color=c, linewidth=0.8, linestyle="--",
                   label=f"mean {np.nanmean(ratio):.3f}")
        ax.fill_between(stats["t"], 0, ratio, alpha=0.12, color=c)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("eig_min / eig_max")
        ax.set_title(f"{label} — spatial degeneracy score  (0=degenerate, 1=rich)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Feature Spatial Distribution — Eigenvalue Ratio of Per-Frame Covariance\n"
        "Low ratio = points lie on a flat surface (corridor). "
        "High ratio = points spread in 3D (feature-rich).",
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, out / "feature_degeneracy.png")

    # ---- CSV export ----
    for label, stats, _ in valid:
        if not stats:
            continue
        slug = label.lower().replace(" / ", "_").replace(" ", "_")
        csv_path = out / f"feature_geometry_{slug}.csv"
        np.savetxt(
            csv_path,
            np.column_stack([stats["t"], stats["count"], stats["eig_ratio"], stats["eig_min"]]),
            delimiter=",",
            header="t,count,eig_ratio,eig_min",
            comments="",
        )
        print(f"  saved: {csv_path}")


# ---------------------------------------------------------------------------
# Summary CSV export
# ---------------------------------------------------------------------------

def export_odom_csv(odom: dict, label: str, out: Path):
    if not odom:
        return
    path = out / f"{label}.csv"
    header = "t,x,y,z,roll_deg,pitch_deg,yaw_deg"
    data = np.column_stack([odom["t"], odom["x"], odom["y"], odom["z"],
                            odom["roll"], odom["pitch"], odom["yaw"]])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"  saved: {path}")


def export_stats_csv(stats_data: list, out: Path):
    if not stats_data:
        return
    t0 = stats_data[0][0]
    fields = [
        "laser_cloud_surf_from_map_num", "laser_cloud_corner_from_map_num",
        "laser_cloud_surf_stack_num", "laser_cloud_corner_stack_num",
        "total_translation", "total_rotation", "translation_from_last", "rotation_from_last",
        "time_elapsed", "latency", "n_iterations", "average_distance",
        "uncertainty_x", "uncertainty_y", "uncertainty_z",
        "uncertainty_roll", "uncertainty_pitch", "uncertainty_yaw",
        "plane_match_success", "plane_no_enough_neighbor", "plane_neighbor_too_far",
        "plane_badpca_structure", "plane_invalid_numerical", "plane_mse_too_large",
        "plane_unknown", "prediction_source",
        "edge_residual_rms", "plane_residual_rms",
        "edge_assoc_dist_mean", "edge_assoc_dist_max",
        "plane_assoc_dist_mean", "plane_assoc_dist_max",
        "hessian_pos_condition_num", "hessian_ori_condition_num",
        "hessian_var_x", "hessian_var_y", "hessian_var_z",
        "hessian_var_roll", "hessian_var_pitch", "hessian_var_yaw",
    ]
    rows = []
    for stamp, msg in stats_data:
        row = [stamp - t0] + [getattr(msg, f, float("nan")) for f in fields]
        rows.append(row)
    path = out / "optimisation_stats.csv"
    header = "t," + ",".join(fields)
    np.savetxt(path, rows, delimiter=",", header=header, comments="")
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse a super_odometry rosbag2 recording.")
    parser.add_argument("-b", "--bag",    required=True, help="Path to the bag directory")
    parser.add_argument("-o", "--output", default="",    help="Output directory for plots/CSVs")
    parser.add_argument("-p", "--prefix", default="",    help="Topic prefix (PROJECT_NAME), e.g. /super_odometry")
    parser.add_argument("--rpe-delta",    type=float, default=0.3,
                        help="Segment length in metres for RPE consistency metric (default: 0.5)")
    parser.add_argument("--map-consistency", action="store_true", default=True,
                        help="Enable map consistency metric (submap ICP). Requires open3d. "
                             "Only meaningful when the robot completes a loop.")
    parser.add_argument("--loop-x-tol",  type=float, default=0.3,
                        help="X-axis tolerance (m) for detecting the loop return point (default: 1.0). "
                             "Y is ignored — X is used because Y can vary with human/odometry error.")
    parser.add_argument("--loop-min-t",  type=float, default=200.0,
                        help="Minimum time (s) between consecutive loop return detections. "
                             "Set this to ~80%% of your expected single-loop duration so that "
                             "slow passes through the start zone don't double-count. (default: 30)")
    parser.add_argument("--icp-max-dist", type=float, default=0.5,
                        help="Max ICP correspondence distance in metres (default: 0.5)")
    parser.add_argument("--mme",          action="store_true", default=True,
                        help="Also compute Map Mean Entropy on the merged map")
    parser.add_argument("--mme-max-pts", type=int, default=30_000,
                        help="Max points used for MME computation — subsampled randomly if exceeded "
                             "(default: 30000)")
    parser.add_argument("--save-map", action="store_true", default=True,
                        help="Save the global map (last laser_cloud_map message) as map.pcd. "
                             "Requires open3d.")
    args = parser.parse_args()

    bag_path = args.bag
    prefix   = args.prefix.rstrip("/")

    if args.output:
        out = Path(args.output)
    else:
        out = Path(bag_path).parent / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")

    # Topic names
    T = {
        "laser_odom":      f"{prefix}/laser_odometry",
        "imu_odom":        f"{prefix}/state_estimation",
        "vio_pred":        f"{prefix}/vio_prediction",
        "lio_pred":        f"{prefix}/lio_prediction",
        "laser_path":      f"{prefix}/laser_odom_path",
        "imu_path":        f"{prefix}/imuodom_path",
        "stats":           f"{prefix}/super_odometry_stats",
        "registered_scan": f"{prefix}/registered_scan",
        "laser_cloud_map": f"{prefix}/laser_cloud_map",
        "feature_info":    f"{prefix}/feature_info",
    }

    # =========================================================================
    # Pass 1 — lightweight topics streamed in one bag pass.
    #
    # Subscribers extract only scalar / array data on the fly and discard each
    # raw message immediately after handling it.  No raw messages are retained
    # after this pass completes.
    # =========================================================================
    print(f"\n[Pass 1] Streaming lightweight topics: {bag_path}")

    odom_sub   = {k: OdomSubscriber() for k in ("laser_odom", "imu_odom", "vio_pred", "lio_pred")}
    path_sub   = {k: PathSubscriber() for k in ("laser_path", "imu_path")}
    stats_sub  = StatsSubscriber()
    feat_sub   = FeatureGeometrySubscriber()

    p1 = BagProcessor(bag_path)
    for k, sub in odom_sub.items():
        p1.subscribe(T[k], sub)
    for k, sub in path_sub.items():
        p1.subscribe(T[k], sub)
    p1.subscribe(T["stats"],        stats_sub)
    p1.subscribe(T["feature_info"], feat_sub)
    p1.spin()

    odom_laser    = odom_sub["laser_odom"].result()
    odom_imu      = odom_sub["imu_odom"].result()
    odom_vio      = odom_sub["vio_pred"].result()
    odom_lio      = odom_sub["lio_pred"].result()
    laser_path_xy = path_sub["laser_path"].result()
    stats_data    = stats_sub.data
    feature_stats = feat_sub.result()
    # subscribers are no longer needed — release raw message refs
    del p1, odom_sub, path_sub, stats_sub, feat_sub

    print(f"  laser_odom    : {len(odom_laser.get('t', []))} poses")
    print(f"  imu_odom      : {len(odom_imu.get('t', []))} poses")
    print(f"  stats         : {len(stats_data)} msgs")
    print(f"  feature_info  : {sum(v['count'].shape[0] for v in feature_stats.values() if 'count' in v)} frames")

    print("\nGenerating plots (pass 1)...")
    odoms_all = {
        "laser_odometry":   odom_laser,
        "state_estimation": odom_imu,
        "vio_prediction":   odom_vio,
        "lio_prediction":   odom_lio,
    }
    plot_trajectory(odoms_all, laser_path_xy, out)
    plot_xyz_vs_time(odoms_all, out)
    plot_rpy_vs_time(odoms_all, out)
    plot_optimisation_stats(stats_data, out)
    plot_rotation_vs_time(stats_data, out)
    plot_residuals(stats_data, out)
    plot_hessian_constraint_quality(stats_data, out)
    plot_association_distances(stats_data, out)
    plot_imu_vs_lidar(odom_laser, odom_imu, out)
    plot_uncertainty_with_trajectory(stats_data, odom_laser, out)
    plot_prediction_source_heatmap(stats_data, odom_laser, out)
    plot_inter_source_drift(
        odom_laser, odom_imu,
        ref_label="laser_odometry", est_label="state_estimation",
        out=out, delta_m=args.rpe_delta,
    )
    plot_loop_drift(odom_laser=odom_laser, out=out,
                    x_tol=args.loop_x_tol, min_loop_t=args.loop_min_t)

    if feature_stats:
        plot_feature_geometry(feature_stats, out)
    else:
        print("  [skip] feature geometry — no /feature_info messages in bag")

    print("\nExporting CSVs...")
    export_odom_csv(odom_laser, "laser_odometry", out)
    export_odom_csv(odom_imu,   "state_estimation", out)
    export_stats_csv(stats_data, out)

    # =========================================================================
    # Pass 2 — point cloud topics, streamed with submap-aware subscribers.
    #
    # Loop boundaries are computed from pass-1 odometry so scans are bucketed
    # into submaps on the fly and voxel-downsampled every FLUSH_EVERY frames.
    # At most one submap's worth of raw points is ever live in memory.
    # =========================================================================

    # Detect loop boundaries from pass-1 odometry before opening the bag again.
    loop_times = _find_all_loop_return_times(
        odom_laser, x_tol=args.loop_x_tol, min_loop_t=args.loop_min_t
    )
    n_submaps   = len(loop_times) + 1
    loop_labels = [chr(ord("A") + i) for i in range(n_submaps)]

    # Absolute-time boundaries for SubmapBuilderSubscriber.
    # odom_laser["t"] is zeroed; t0_abs reconstructs absolute ROS time.
    # Peek at the first laser_odom message to recover the absolute ROS timestamp
    # that corresponds to odom_laser["t"][0] == 0.  We need this to convert
    # relative loop_times back to absolute boundaries for SubmapBuilderSubscriber.
    _t0_reader = open_bag(bag_path)
    _t0_types  = topic_type_map(_t0_reader)
    _t0_filter = rosbag2_py.StorageFilter(topics=[T["laser_odom"]])
    _t0_reader.set_filter(_t0_filter)
    t0_abs = 0.0
    if _t0_reader.has_next() and T["laser_odom"] in _t0_types:
        _topic, _raw, _stamp_ns = _t0_reader.read_next()
        _msg = deserialize_message(_raw, get_message(_t0_types[T["laser_odom"]]))
        if hasattr(_msg, "header"):
            t0_abs = _msg.header.stamp.sec + _msg.header.stamp.nanosec * 1e-9
        else:
            t0_abs = _stamp_ns * 1e-9
    del _t0_reader, _t0_types, _t0_filter

    boundaries_abs = (
        [t0_abs - 1.0]
        + [t0_abs + lt for lt in loop_times]
        + [float("inf")]
    )

    print(f"\n[Pass 2] Streaming point cloud topics: {bag_path}")
    print(f"  {n_submaps} submap(s): {', '.join(loop_labels)}")

    submap_sub = SubmapBuilderSubscriber(boundaries_abs, voxel_size=args.icp_max_dist / 2)
    map_sub    = LastMessageSubscriber()

    p2 = BagProcessor(bag_path)
    p2.subscribe(T["registered_scan"], submap_sub)
    if args.save_map:
        p2.subscribe(T["laser_cloud_map"], map_sub)
    p2.spin()

    submaps_pts = submap_sub.result()
    del p2, submap_sub

    # ---- save global map ----
    if args.save_map:
        save_global_map_pcd(map_sub.msg, out)
    del map_sub

    # ---- map consistency ICP ----
    if args.map_consistency:
        if loop_times:
            print(f"\n  Map consistency: {n_submaps} submap(s) — {', '.join(loop_labels)}")
            plot_map_consistency(
                submaps_pts=submaps_pts,
                loop_labels=loop_labels,
                loop_times=loop_times,
                odom_laser=odom_laser,
                out=out,
                compute_mme=args.mme,
                mme_max_pts=args.mme_max_pts,
                icp_max_dist=args.icp_max_dist,
            )
        else:
            print("  [skip] map consistency — no loop returns detected "
                  f"(x_tol={args.loop_x_tol} m, min_loop_t={args.loop_min_t} s). "
                  "Try --loop-x-tol or --loop-min-t to relax the criteria.")

    print(f"\nDone. All outputs in: {out}")


if __name__ == "__main__":
    main()
