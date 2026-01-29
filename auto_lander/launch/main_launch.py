from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # ----- Create tf frames -----
    # Static transform from "base_link" to "camera_link"
    base_to_camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        output='screen',
        arguments=[
            '0.0', '0.0', '-0.1249',                # x, y, z translation
            '-1.570796326', '0.0', '3.1415926535',   # roll, pitch, yaw (radians)
            'base_link',                            # parent frame
            'camera_link'                           # child frame
        ]
    )
    
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

    # Run main controller node
    controller = Node(
        package='auto_lander',
        executable='controller',
        name='controller_node',
        output='screen'
    )

    return LaunchDescription([
        base_to_camera_tf_node,
        tag_pose_detector,
        controller
    ])
