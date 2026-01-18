from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
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

    controller = Node(
        package='auto_lander',
        executable='controller',
        name='controller_node',
        output='screen'
    )

    return LaunchDescription([
        tag_pose_detector,
        controller
    ])
