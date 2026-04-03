import os
from datetime import datetime

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
import launch_ros

def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)

def generate_launch_description():
    config_path = get_share_file(
        package_name="super_odometry",
        file_name="config/os1_128.yaml")
    calib_path = get_share_file(
        package_name="super_odometry",
        file_name="config/ouster/os1_128_calibration.yaml"
    )
    home_directory = os.path.expanduser("~")
    
    config_path_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_path,
        description="Path to config file for super_odometry"
    )
    calib_path_arg = DeclareLaunchArgument(
        "calibration_file",
        default_value=calib_path,
    )
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="integrated_to_init"
    )
    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="map",
    )
    world_frame_rot_arg = DeclareLaunchArgument(
        "world_frame_rot",
        default_value="map_rot",
    )
    sensor_frame_arg = DeclareLaunchArgument(
        "sensor_frame",
        default_value="sensor",
    )
    sensor_frame_rot_arg = DeclareLaunchArgument(
        "sensor_frame_rot",
        default_value="sensor_rot",
    )
    record_arg = DeclareLaunchArgument(
        "record",
        default_value="false",
        description="Set to true to record all SuperOdometry output topics to a bag",
    )
    output_dir_arg = DeclareLaunchArgument(
        "output_dir",
        default_value=os.path.join(os.path.expanduser("~"), "bags"),
        description="Parent directory where the bag folder will be created",
    )
    bag_name_arg = DeclareLaunchArgument(
        "bag_name",
        default_value=f"super_odom_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        description="Bag folder name (created inside output_dir)",
    )

    record_process = TimerAction(
        period=3.0,  # wait for publishers to come up before subscribing
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
                # FeatureExtraction
                "/velodyne_cloud_2",
                "/feature_info",
                "/bob_points",
                "/planner_points",
                "/edge_points",
                # LaserMapping
                "/laser_cloud_surround",
                "/laser_cloud_map",
                "/overall_map",
                "/registered_scan",
                "/laser_odometry",
                "/aft_mapped_to_init_incremental",
                "/vio_prediction",
                "/lio_prediction",
                "/laser_odom_path",
                "/super_odometry_stats",
                "/prediction_source",
                # ImuPreintegration
                "/state_estimation",
                "/state_estimation_health",
                "/imuodom_path",
                # LidarSlam uncertainties
                "/uncertainty_X",
                "/uncertainty_Y",
                "/uncertainty_Z",
                "/uncertainty_roll",
                "/uncertainty_pitch",
                "/uncertainty_yaw",
            ],
            output="screen",
        )],
    )

    feature_extraction_node = Node(
        package="super_odometry",
        executable="feature_extraction_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file"),
            { "calibration_file": LaunchConfiguration("calibration_file"),
        }],
    )

    laser_mapping_node = Node(
        package="super_odometry",
        executable="laser_mapping_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file"),
            { "calibration_file": LaunchConfiguration("calibration_file"),
             "map_dir": os.path.join(home_directory, "lidar_maps"),
        }],
        remappings=[
            ("laser_odom_to_init", LaunchConfiguration("odom_topic")),
        ]
    )

    imu_preintegration_node = Node(
        package="super_odometry",
        executable="imu_preintegration_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file"),
            { "calibration_file": LaunchConfiguration("calibration_file")
        }],
    )

    # Static TF: os_lidar -> os_imu (from calibration: imu^T_laser = [-0.006253, 0.011775, -0.007645])
    # IMU position in lidar frame = -T = [0.006253, -0.011775, 0.007645]. Rotation is identity.
    # static_tf_lidar_imu = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="ouster_lidar_imu_tf",
    #     arguments=["0.006253", "-0.011775", "0.007645", "0", "0", "0", "os_lidar", "os_imu"],
    # )

    
    return LaunchDescription([
        launch_ros.actions.SetParameter(name='use_sim_time', value='false'),
        config_path_arg,
        calib_path_arg,
        odom_topic_arg,
        world_frame_arg,
        world_frame_rot_arg,
        sensor_frame_arg,
        sensor_frame_rot_arg,
        record_arg,
        output_dir_arg,
        bag_name_arg,
        feature_extraction_node,
        laser_mapping_node,
        imu_preintegration_node,
        # static_tf_lidar_imu,
        record_process,
    ])
