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
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rpy(qx, qy, qz, qw):
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
        r, p_, yw = quat_to_rpy(o.x, o.y, o.z, o.w)
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


def _build_submap(scan_msgs: list, t_lo: float, t_hi: float) -> np.ndarray:
    """Merge registered_scan PointCloud2 messages whose stamp falls in [t_lo, t_hi)."""
    chunks = []
    for stamp, msg in scan_msgs:
        if t_lo <= stamp < t_hi:
            xyz = _pc2_to_xyz(msg)
            if xyz.size:
                chunks.append(xyz)
    return np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float32)


def _find_loop_return_time(odom: dict, x_tol: float, min_loop_t: float) -> float | None:
    """
    Find the first time (relative seconds) the robot returns to its starting
    X coordinate within `x_tol` metres, after at least `min_loop_t` seconds.

    X is used as the primary indicator because the user notes Y can vary due
    to human/odometry variation but X should be consistent at the loop return.
    """
    if not odom:
        return None
    x0 = odom["x"][0]
    for t, x in zip(odom["t"], odom["x"]):
        if t < min_loop_t:
            continue
        if abs(x - x0) <= x_tol:
            return float(t)
    return None


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


def save_global_map_pcd(map_msgs: list, out: Path) -> None:
    """
    Save the global accumulated map as a PCD file.

    Uses the last laser_cloud_map message, which contains the most complete
    version of the map.  Requires open3d.
    """
    if not HAS_O3D:
        print("  [skip] save map — open3d not installed (pip install open3d)")
        return
    if not map_msgs:
        print("  [skip] save map — no laser_cloud_map messages in bag")
        return

    _, msg = map_msgs[-1]
    pts = _pc2_to_xyz(msg)
    if pts.size == 0:
        print("  [skip] save map — last laser_cloud_map message is empty")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    pcd_path = out / "map.pcd"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"  saved: {pcd_path}  ({len(pcd.points):,} points)")


def plot_map_consistency(scan_data: list, odom_laser: dict, out: Path,
                         x_tol: float = 1.0, min_loop_t: float = 30.0,
                         compute_mme: bool = False,
                         mme_max_pts: int = 30_000) -> None:
    """
    Map consistency metric (optional — requires open3d).

    1. Detects the loop return time: first moment the robot's X coordinate
       returns to within `x_tol` of its starting X, after `min_loop_t` seconds.
       Y is ignored because human/odometry variation makes it unreliable.
    2. Splits /registered_scan messages at that time into submap A (first pass)
       and submap B (second pass).
    3. Runs point-to-point ICP between the submaps.  If SLAM is working well
       the submaps are already aligned, so ICP should converge with a near-
       identity transform and low inlier RMSE.
    4. Reports fitness score, inlier RMSE, and — if requested — MME on the
       merged map.
    """
    if not HAS_O3D:
        print("  [skip] map consistency — open3d not installed (pip install open3d)")
        return
    if not scan_data:
        print("  [skip] map consistency — no /registered_scan messages in bag")
        return
    if not odom_laser:
        print("  [skip] map consistency — no laser_odometry data to detect loop")
        return

    # Timestamps in the scan data are absolute; align with odom relative time
    t0_abs = scan_data[0][0] - odom_laser["t"][0] if odom_laser["t"][0] != 0 else scan_data[0][0]

    loop_t_rel = _find_loop_return_time(odom_laser, x_tol=x_tol, min_loop_t=min_loop_t)
    if loop_t_rel is None:
        print(f"  [skip] map consistency — robot never returned to starting X "
              f"(±{x_tol} m) after {min_loop_t} s. "
              f"Try --loop-x-tol or --loop-min-t to relax the criteria.")
        return

    # Convert loop time from odom-relative to absolute stamp space
    t_start = scan_data[0][0]
    t_split  = t_start + loop_t_rel
    t_end    = scan_data[-1][0]

    print(f"\n  Map consistency: loop return detected at t={loop_t_rel:.1f} s "
          f"(x ≈ {odom_laser['x'][0]:.2f} m ± {x_tol} m)")
    print(f"    Submap A: {t_start:.1f} → {t_split:.1f} s")
    print(f"    Submap B: {t_split:.1f} → {t_end:.1f} s")

    pts_a = _build_submap(scan_data, t_start, t_split)
    pts_b = _build_submap(scan_data, t_split, t_end + 1.0)

    if len(pts_a) < 100 or len(pts_b) < 100:
        print(f"  [skip] map consistency — submaps too sparse "
              f"(A={len(pts_a)}, B={len(pts_b)} pts). "
              f"Check that /registered_scan was recorded.")
        return

    print(f"    Submap A: {len(pts_a):,} pts  |  Submap B: {len(pts_b):,} pts")

    # Build Open3D point clouds (voxel-downsample for speed)
    def _to_o3d(pts: np.ndarray, voxel: float = 0.1) -> "o3d.geometry.PointCloud":
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return pc.voxel_down_sample(voxel)

    pcd_a = _to_o3d(pts_a)
    pcd_b = _to_o3d(pts_b)

    # Coarse initialisation from odometry drift.
    # At t_split the robot has physically returned to the start, but SLAM has
    # accumulated error so its reported pose has drifted.  The pose at t_split
    # IS the drift: submap B needs to be shifted by -drift to sit on top of A.
    # Find the odom sample closest to t_split and read its pose.
    odom_t = np.asarray(odom_laser["t"])
    idx_split = int(np.argmin(np.abs(odom_t - loop_t_rel)))
    drift_x   = float(odom_laser["x"][idx_split] - odom_laser["x"][0])
    drift_y   = float(odom_laser["y"][idx_split] - odom_laser["y"][0])
    drift_z   = float(odom_laser["z"][idx_split] - odom_laser["z"][0])

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

    R0     = _rpy_to_rot(odom_laser["roll"][0],          odom_laser["pitch"][0],          odom_laser["yaw"][0])
    R_split = _rpy_to_rot(odom_laser["roll"][idx_split], odom_laser["pitch"][idx_split],  odom_laser["yaw"][idx_split])
    # Relative rotation: how much has orientation drifted
    R_drift = R0.T @ R_split

    T_init = np.eye(4)
    T_init[:3, :3] = R_drift.T          # inverse: bring B back to A's orientation
    T_init[:3,  3] = [-drift_x, -drift_y, -drift_z]

    print(f"\n  Odometry-derived initial drift: "
          f"Δxyz=({drift_x:.2f}, {drift_y:.2f}, {drift_z:.2f}) m  "
          f"|t|={np.linalg.norm([drift_x, drift_y, drift_z]):.2f} m")

    # ICP: refine the coarse odometry correction.  With the initial transform
    # bringing B within ~1 m of A, a tight correspondence distance is reliable.
    print(f"  Running point-to-point ICP (B → A, coarse-initialised) …")
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_b, pcd_a,
        max_correspondence_distance=0.5,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    T = icp_result.transformation
    t_drift = float(np.linalg.norm(T[:3, 3]))
    angle_drift = float(np.degrees(np.arccos(
        np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    )))
    print(f"    ICP fitness          : {icp_result.fitness:.4f}  "
          f"(1.0 = all points matched)")
    print(f"    ICP inlier RMSE      : {icp_result.inlier_rmse:.4f} m")
    print(f"    Drift translation    : {t_drift:.4f} m")
    print(f"    Drift rotation       : {angle_drift:.3f} deg")

    pcd_b_aligned = pcd_b.transform(T)

    # NN distance on the ICP-aligned B: measures surface consistency, not drift.
    print(f"\n  Computing NN distances (aligned B → A) …")
    dists   = np.asarray(pcd_b_aligned.compute_point_cloud_distance(pcd_a))
    rmse    = float(np.sqrt(np.mean(dists ** 2)))
    mean_nn = float(np.mean(dists))
    p95     = float(np.percentile(dists, 95))

    print(f"    RMSE    : {rmse:.4f} m")
    print(f"    mean NN : {mean_nn:.4f} m")
    print(f"    p95 NN  : {p95:.4f} m")

    mme_val = float("nan")
    if compute_mme:
        merged = np.vstack([pts_a, pts_b])
        print(f"    Computing MME on {len(merged):,} merged points "
              f"(capped at {mme_max_pts:,}) …")
        mme_val = _compute_mme(merged, max_pts=mme_max_pts)
        print(f"    MME           : {mme_val:.4f} (lower = sharper map)")

    # --- save stats text ---
    stats_file = out / "map_consistency.txt"
    with open(stats_file, "w") as f:
        f.write("Map consistency — ICP alignment then nearest-neighbour (B → A)\n")
        f.write(f"Loop return detected at: {loop_t_rel:.2f} s (x_tol={x_tol} m, "
                f"min_loop_t={min_loop_t} s)\n")
        f.write(f"Submap A points (downsampled): {len(pcd_a.points)}\n")
        f.write(f"Submap B points (downsampled): {len(pcd_b.points)}\n\n")
        f.write(f"ICP fitness          : {icp_result.fitness:.6f}\n")
        f.write(f"ICP inlier RMSE      : {icp_result.inlier_rmse:.6f} m\n")
        f.write(f"Drift translation    : {t_drift:.6f} m\n")
        f.write(f"Drift rotation       : {angle_drift:.4f} deg\n\n")
        f.write(f"Post-alignment RMSE  : {rmse:.6f} m\n")
        f.write(f"Post-alignment mean NN: {mean_nn:.6f} m\n")
        f.write(f"Post-alignment P95 NN: {p95:.6f} m\n")
        if compute_mme:
            f.write(f"MME (merged map)     : {mme_val:.6f}\n")
    print(f"  saved: {stats_file}")

    # --- figure ---
    pts_b_aligned = np.asarray(pcd_b_aligned.points)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # Top-left: submap A top-down
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(pts_a[::10, 0], pts_a[::10, 1], s=0.3, c="tab:blue", alpha=0.4)
    ax.set_title("Submap A — first pass (XY)")
    ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Top-right: aligned submap B top-down
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(pts_b_aligned[::10, 0], pts_b_aligned[::10, 1], s=0.3, c="tab:orange", alpha=0.4)
    ax.set_title(f"Submap B — second pass, ICP-aligned (drift {t_drift:.3f} m / {angle_drift:.2f}°)")
    ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Bottom-left: overlay after alignment
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(pts_a[::10, 0], pts_a[::10, 1], s=0.3, c="tab:blue",   alpha=0.35, label="A")
    ax.scatter(pts_b_aligned[::10, 0], pts_b_aligned[::10, 1], s=0.3, c="tab:orange", alpha=0.35, label="B (aligned)")
    ax.set_title("Overlay after ICP alignment")
    ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(fontsize=8, markerscale=5)

    # Bottom-right: aligned B coloured by NN distance to A
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(pts_b_aligned[::10, 0], pts_b_aligned[::10, 1],
                    s=0.3, c=dists[::10], cmap="hot_r",
                    vmin=0, vmax=np.percentile(dists, 99), alpha=0.6)
    plt.colorbar(sc, ax=ax, label="NN dist (m)")
    ax.set_title(f"B (aligned) coloured by NN dist to A  (RMSE={rmse:.3f} m)")
    ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    fig.suptitle(
        f"Map Consistency — ICP drift: {t_drift:.3f} m / {angle_drift:.2f}°  |  "
        f"Post-alignment NN RMSE: {rmse:.3f} m",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, out / "map_consistency.png")


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
    parser.add_argument("--loop-x-tol",  type=float, default=0.5,
                        help="X-axis tolerance (m) for detecting the loop return point (default: 1.0). "
                             "Y is ignored — X is used because Y can vary with human/odometry error.")
    parser.add_argument("--loop-min-t",  type=float, default=336.0,
                        help="Minimum elapsed time (s) before a loop return is considered (default: 30)")
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
        "laser_odom":     f"{prefix}/laser_odometry",
        "imu_odom":       f"{prefix}/state_estimation",
        "vio_pred":       f"{prefix}/vio_prediction",
        "lio_pred":       f"{prefix}/lio_prediction",
        "laser_path":     f"{prefix}/laser_odom_path",
        "imu_path":       f"{prefix}/imuodom_path",
        "stats":          f"{prefix}/super_odometry_stats",
        "registered_scan":  f"{prefix}/registered_scan",
        "laser_cloud_map":  f"{prefix}/laser_cloud_map",
    }

    print(f"\nReading bag: {bag_path}")
    data = read_messages(bag_path, list(T.values()))

    # Extract odometry dicts
    odom_laser = extract_odom(data[T["laser_odom"]])
    odom_imu   = extract_odom(data[T["imu_odom"]])
    odom_vio   = extract_odom(data[T["vio_pred"]])
    odom_lio   = extract_odom(data[T["lio_pred"]])

    # Extract paths
    laser_path_xy = extract_path(data[T["laser_path"]])
    imu_path_xy   = extract_path(data[T["imu_path"]])

    stats_data = data[T["stats"]]

    print(f"\n  Messages received:")
    for key, topic in T.items():
        print(f"    {key:12s} ({topic}): {len(data[topic])} msgs")

    print("\nGenerating plots...")

    odoms_all = {
        "laser_odometry":    odom_laser,
        "state_estimation":  odom_imu,
        "vio_prediction":    odom_vio,
        "lio_prediction":    odom_lio,
    }

    plot_trajectory(odoms_all, laser_path_xy, out)
    plot_xyz_vs_time(odoms_all, out)
    plot_rpy_vs_time(odoms_all, out)
    plot_optimisation_stats(stats_data, out)
    plot_rotation_vs_time(stats_data, out)
    plot_imu_vs_lidar(odom_laser, odom_imu, out)
    plot_inter_source_drift(
        odom_laser, odom_imu,
        ref_label="laser_odometry", est_label="state_estimation",
        out=out, delta_m=args.rpe_delta,
    )

    if args.save_map:
        save_global_map_pcd(data[T["laser_cloud_map"]], out)

    if args.map_consistency:
        plot_map_consistency(
            scan_data=data[T["registered_scan"]],
            odom_laser=odom_laser,
            out=out,
            x_tol=args.loop_x_tol,
            min_loop_t=args.loop_min_t,
            compute_mme=args.mme,
            mme_max_pts=args.mme_max_pts,
        )

    print("\nExporting CSVs...")
    export_odom_csv(odom_laser, "laser_odometry", out)
    export_odom_csv(odom_imu,   "state_estimation", out)
    export_stats_csv(stats_data, out)

    print(f"\nDone. All outputs in: {out}")


if __name__ == "__main__":
    main()
