from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():

    # ----- Step 1a: Gazebo Camera + Gimbal Bridge -----
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/tilt_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/model/LandingVehicle/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/model/iris_with_gimbal/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/cmd_rover_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "--ros-args",
            "-r",
            "/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/tilt_link/sensor/camera/image:=/camera/image_raw",
            "-r",
            "/model/LandingVehicle/odometry:=/landing_pad/odom",
            "-r",
            "/model/iris_with_gimbal/odometry:=/quadcopter/true_odom",
        ],
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
    apriltag_pose_detector = Node(
        package="auto_lander",
        executable="apriltag",
        name="apriltag_node",
        output="screen",
        sigterm_timeout="20",
        sigkill_timeout="30",
        parameters=[
            {"diagnostics_enabled": True},
            {"image_source": "topic"},
            {"show_debug_window": True},
            {"enable_debug_publish": False},
            {"create_video": False},
            {"use_sim_time": True},
        ],
    )
    
    # Run target pose detector node
    yolo_pose_detector = Node(
        package="auto_lander",
        executable="yolo",
        name="apriltag_node",
        output="screen",
        sigterm_timeout="20",
        sigkill_timeout="30",
        parameters=[
            {"diagnostics_enabled": True},
            {"image_source": "topic"},
            {"show_debug_window": True},
            {"enable_debug_publish": False},
            {"create_video": False},
            {"use_sim_time": True},
        ],
    )

    # Run main controller node
    controller = Node(
        package="auto_lander",
        executable="controller",
        name="controller_node",
        output="screen",
        parameters=[
            {"diagnostics_enabled": True},
            {"ground_truth_available": True},
            {"use_sim_time": True},
        ],
    )

    # Run AGV controller node (SITL only)
    agv_controller = Node(
        package="auto_lander",
        executable="agv_controller",
        name="agv_controller",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    delayed_agv_controller = TimerAction(
        period=25.0,  # Wait a few secs
        actions=[agv_controller],
    )

    return LaunchDescription(
        [
            gz_bridge,
            # base_to_camera_tf_node,
            apriltag_pose_detector,
            yolo_pose_detector,
            controller,
            delayed_agv_controller,
        ]
    )
