from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='auto_lander',
            executable='bearing_measurement_generation',
            name='yolo_image_node',
            output='screen',
            parameters=[
                {'image_source': 'topic'},
                {'show_debug_window': True},
                {'enable_debug_publish': True}
            ]
        ),
        Node(
            package='auto_lander',
            executable='controller',
            name='controller_node',
            output='screen'
        )
    ])
