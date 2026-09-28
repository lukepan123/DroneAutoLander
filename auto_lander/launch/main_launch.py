import os
from datetime import datetime

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess

def generate_launch_description():

    # One ID shared by the bag and the controller's CSV/event/truth files
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = os.path.join(os.getcwd(), "bags", run_id)

    # ----- Step 1a: Rosbag recorder -----
    # bag_record = ExecuteProcess(
    #     cmd=[
    #         "ros2", "bag", "record",
    #         "-s", "mcap",
    #         "--use-sim-time",
    #         "-o", bag_dir,
    #         "/tf", "/tf_static", "/clock",
    #         "/mavros/state",
    #         "/mavros/global_position/local",
    #         "/mavros/global_position/global",
    #         "/mavros/imu/data",
    #         "/mavros/setpoint_raw/attitude",
    #         "/mavros/setpoint_velocity/cmd_vel",
    #         "/mavros/setpoint_raw/global",
    #         "/landing_pad/found",
    #         "/landing_pad/pipeline_timing",
    #         "/landing_pad/yolo_pipeline_timing",
    #         "/quadcopter/true_odom",
    #         "/landing_pad/odom",
    #     ],
    #     output="screen",
    #     sigterm_timeout="20",   # give it time to finalise the bag on Ctrl+C
    #     sigkill_timeout="30",
    # )

    # ----- Step 2a: Gazebo Camera + Gimbal Bridge -----
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/tilt_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/landing_pad/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/model/iris_with_gimbal/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",

            "/cmd_rover_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",

            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",

            "--ros-args",
            "-r",
            "/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/tilt_link/sensor/camera/image:=/camera/image_raw",
            "-r",
            "/model/iris_with_gimbal/odometry:=/quadcopter/true_odom",
        ],
    )

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
            {"create_video": True},
            {"use_sim_time": True},
        ],
    )
    
    # Run target pose detector node
    yolo_pose_detector = Node(
        package="auto_lander",
        executable="yolo",
        name="yolo_node",
        output="screen",
        sigterm_timeout="20",
        sigkill_timeout="30",
        parameters=[
            {"diagnostics_enabled": True},
            {"image_source": "topic"},
            {"show_debug_window": True},
            {"enable_debug_publish": False},
            {"create_video": True},
            {"use_sim_time": True},
            {"ground_z": 1.5},
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
            {"run_id": run_id},
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
            #bag_record,
            gz_bridge,
            apriltag_pose_detector,
            yolo_pose_detector,
            controller,
            delayed_agv_controller,
        ]
    )
