import os
from datetime import datetime

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import launch_ros


def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)


def generate_launch_description():

    # ------------------------------------------------------------------ paths
    so_config = get_share_file("super_odometry", "config/os1_128.yaml")
    so_calib  = get_share_file("super_odometry", "config/ouster/os1_128_calibration.yaml")
    ov_config = os.path.join(
        get_package_share_directory("ov_msckf"), "config", "slam_fusion", "estimator_config.yaml"
    )
    home = os.path.expanduser("~")

    # --------------------------------------------------------- launch arguments
    so_config_arg  = DeclareLaunchArgument("config_file",       default_value=so_config)
    so_calib_arg   = DeclareLaunchArgument("calibration_file",  default_value=so_calib)
    odom_topic_arg = DeclareLaunchArgument("odom_topic",        default_value="integrated_to_init")
    world_frame_arg        = DeclareLaunchArgument("world_frame",        default_value="map")
    world_frame_rot_arg    = DeclareLaunchArgument("world_frame_rot",    default_value="map_rot")
    sensor_frame_arg       = DeclareLaunchArgument("sensor_frame",       default_value="sensor")
    sensor_frame_rot_arg   = DeclareLaunchArgument("sensor_frame_rot",   default_value="sensor_rot")

    ov_config_arg = DeclareLaunchArgument(
        "ov_config_path",
        default_value=ov_config,
        description="Path to OpenVINS estimator_config.yaml"
    )
    ov_enable_arg = DeclareLaunchArgument(
        "ov_enable", default_value="true",
        description="Set to false to run SuperOdom without VIO (IMU-only fallback)"
    )
    record_arg = DeclareLaunchArgument(
        "record", default_value="false",
        description="Record all output topics to a bag"
    )
    output_dir_arg = DeclareLaunchArgument(
        "output_dir",
        default_value=os.path.join(home, "bags"),
    )
    bag_name_arg = DeclareLaunchArgument(
        "bag_name",
        default_value=f"super_odom_vins_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    # ------------------------------------------------------------- record bag
    record_process = TimerAction(
        period=3.0,
        condition=IfCondition(LaunchConfiguration("record")),
        actions=[ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "--output", PathJoinSubstitution([
                    LaunchConfiguration("output_dir"),
                    LaunchConfiguration("bag_name"),
                ]),
                "--compression-mode", "file",
                "--compression-format", "zstd",
                # SuperOdom outputs
                "/super_odometry_stats",
                "/laser_odometry",
                "/laser_odom_path",
                "/state_estimation",
                "/imuodom_path",
                "/registered_scan",
                "/vio_prediction",
                "/prediction_source",
                # Camera
                "/oak/rgb/image_raw/compressed",
                # OpenVINS outputs
                "/ov_msckf/odomimu",
                "/ov_msckf/pathimu",
                "/ov_msckf/points_slam",
            ],
            output="screen",
        )],
    )

    # --------------------------------------------------------- SuperOdom nodes
    feature_extraction_node = Node(
        package="super_odometry",
        executable="feature_extraction_node",
        output={"stdout": "screen", "stderr": "screen"},
        parameters=[
            LaunchConfiguration("config_file"),
            {"calibration_file": LaunchConfiguration("calibration_file")},
        ],
    )

    laser_mapping_node = Node(
        package="super_odometry",
        executable="laser_mapping_node",
        output={"stdout": "screen", "stderr": "screen"},
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "calibration_file": LaunchConfiguration("calibration_file"),
                "map_dir": os.path.join(home, "lidar_maps"),
            },
        ],
        remappings=[("laser_odom_to_init", LaunchConfiguration("odom_topic"))],
    )

    imu_preintegration_node = Node(
        package="super_odometry",
        executable="imu_preintegration_node",
        output={"stdout": "screen", "stderr": "screen"},
        parameters=[
            LaunchConfiguration("config_file"),
            {"calibration_file": LaunchConfiguration("calibration_file")},
        ],
    )

    # ----------------------------------------- OAK-D image decompressor
    # image_transport republish: compressed -> raw so OpenVINS gets sensor_msgs/Image
    decompress_node = Node(
        package="image_transport",
        executable="republish",
        name="decompress_oak_rgb",
        arguments=["compressed", "raw"],
        remappings=[
            ("in/compressed", "/oak/rgb/image_raw/compressed"),
            ("out", "/oak/rgb/image_raw"),
        ],
        output="screen",
    )

    # ------------------------------------------------------- OpenVINS node
    # Publishes pose on /ov_msckf/odomimu (nav_msgs/Odometry in IMU world frame)
    # SuperOdom laser_mapping_node subscribes to this topic directly.
    openvins_node = Node(
        package="ov_msckf",
        executable="run_subscribe_msckf",
        namespace="ov_msckf",
        condition=IfCondition(LaunchConfiguration("ov_enable")),
        output={"stdout": "screen", "stderr": "screen"},
        parameters=[
            {"config_path": LaunchConfiguration("ov_config_path")},
            {"verbosity": "INFO"},
            {"use_stereo": False},
            {"max_cameras": 1},
        ],
    )

    return LaunchDescription([
        launch_ros.actions.SetParameter(name="use_sim_time", value="false"),
        so_config_arg, so_calib_arg, odom_topic_arg,
        world_frame_arg, world_frame_rot_arg,
        sensor_frame_arg, sensor_frame_rot_arg,
        ov_config_arg, ov_enable_arg,
        record_arg, output_dir_arg, bag_name_arg,
        decompress_node,
        feature_extraction_node,
        laser_mapping_node,
        imu_preintegration_node,
        openvins_node,
        record_process,
    ])
