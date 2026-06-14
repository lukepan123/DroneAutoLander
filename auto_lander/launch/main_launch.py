from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # ----- Step 1a: Gazebo Camera + Gimbal Bridge -----
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        output='screen',
        arguments=[
            '/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/tilt_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/model/LandingVehicle/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '--ros-args',
            '-r', '/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/tilt_link/sensor/camera/image:=/camera/image_raw',
            '-r', '/model/LandingVehicle/odometry:=/landing_pad/odom'
        ]
    )

    # ----- Create tf frames -----
    # base_to_camera_tf_node = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='base_to_camera_tf',
    #     output='screen',
    #     arguments=[
    #         '0.0', '0.0', '-0.1249',
    #         '-1.570796326', '0.0', '3.1415926535',
    #         'base_link',
    #         'camera_link'
    #     ]
    # )

    # Run target pose detector node
    tag_pose_detector = Node(
        package='auto_lander',
        executable='landing_pad_detector',
        name='image_node',
        output='screen',
        parameters=[
            {'image_source': 'topic'},
            {'show_debug_window': True},
            {'enable_debug_publish': False},
            {'create_video': False},
            {'use_sim_time': True}
        ]
    )

    # Run main controller node
    controller = Node(
        package='auto_lander',
        executable='controller',
        name='controller_node',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )

    return LaunchDescription([
        gz_bridge,
        # base_to_camera_tf_node,
        tag_pose_detector,
        controller
    ])
