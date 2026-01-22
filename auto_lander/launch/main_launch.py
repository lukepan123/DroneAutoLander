from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Run target pose detector node
    tag_pose_detector = Node(
        package='auto_lander',
        executable='tagposedetector',
        name='image_node',
        output='screen',
        parameters=[
            {'image_source': 'topic'},
            {'show_debug_window': True},
            {'enable_debug_publish': False},
            {'create_video': False}
        ]
    )

    # Run target pose EKF filter node
    ekf_params = {
        'frequency': 15.0,
        'sensor_timeout': 0.2,
        'two_d_mode': False,

        'map_frame': 'map',
        'odom_frame': 'odom_target',
        'base_link_frame': 'target',
        'world_frame': 'map',

        'publish_tf': False,

        # -------- ARUCO TAG POSITIONS --------       
        'pose0': '/target_pose',
        'pose0_config': [
            True, True, True,
            False, False, True,
            False, False, False,
            False, False, False
        ],
        'pose0_differential': False,
        'pose0_relative': False,

        # -------- ARUCO TAG VELOCITIES --------  
        'twist0': '/target_twist',
        'twist0_config': [
            False, False, False,   # x y z
            False, False, False,   # roll pitch yaw
            True, True, True,      # vx vy vz
            False, False, False    # vroll vpitch vyaw
        ],
        'twist0_relative': False,
    }

    target_pose_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='target_ekf',
        output='screen',
        parameters=[ekf_params]
    )

    # Run main controller node
    controller = Node(
        package='auto_lander',
        executable='controller',
        name='controller_node',
        output='screen'
    )

    return LaunchDescription([
        target_pose_ekf_node,
        tag_pose_detector,
        controller
    ])
