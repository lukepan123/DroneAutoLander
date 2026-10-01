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

    # Run AprilTag landing pad detector node
    apriltag_pose_detector = Node(
        package="auto_lander",
        executable="apriltag",
        name="apriltag_node",
        output="screen",
        sigterm_timeout="20",
        sigkill_timeout="30",
        parameters=[
            # ----- General -----
            {"diagnostics_enabled": True},
            {"enable_debug_publish": False},
            {"image_source": "topic"},
            {"use_sim_time": True},

            # ----- Webcam -----
            {"webcam_index": 0},
            {"webcam_fps": 30.0},

            # ----- Display / recording -----
            {"show_debug_window": True},
            {"save_frames": False},
            {"create_video": True},
            {"video_fps": 10.0},
            {"output_dir": ""},

            # ----- Processing -----
            {"apriltag_processing_rate": 10.0},
            {"frame_capture_rate": 24.0},
            {"img_width": 960},
            {"img_height": 540},

            # ----- AprilTag detector -----
            {"apriltag_family": "tag36h11"},
            {"apriltag_quad_decimate": 2.0},
            {"apriltag_quad_sigma": 0.0},
            {"apriltag_refine_edges": 1},
            {"apriltag_decode_sharpening": 0.75},
            {"apriltag_debug": 0},

            {"apriltag_tag_ids": [11, 21, 31]},
            {"apriltag_spacing_m": 0.341},
            {"apriltag_main_offset_m": -0.0912 + 0.15},
            {"apriltag_main_tag_size_m": 0.481},
            {"apriltag_small_tag_size_m": 0.072},
            {"apriltag_tag_to_pad_yaw_deg": 90.0},

            # ----- Camera -----
            {"camera_fov_horizontal": 2.7925268},
            {"camera_distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]},

            # ----- Timestamp alignment buffers -----
            {"odometry_buffer_window_s": 1.0},
        ],
    )
    
    # Run YOLO landing pad detector node
    yolo_pose_detector = Node(
        package="auto_lander",
        executable="yolo",
        name="yolo_node",
        output="screen",
        sigterm_timeout="20",
        sigkill_timeout="30",
        parameters=[
            # ----- General -----
            {"diagnostics_enabled": True},
            {"use_sim_time": True},

            # ----- Display / recording -----
            {"show_debug_window": True},
            {"save_frames": False},
            {"create_video": True},
            {"video_fps": 10.0},
            {"output_dir": ""},

            # ----- Processing -----
            {"yolo_processing_rate": 10.0},
            {"img_width": 640},
            {"img_height": 384},

            # ----- YOLO -----
            {"yolo_enabled": True},
            {"yolo_model_path": "ugv_sim_yolo11n_openvino_model"},
            {"yolo_conf_threshold": 0.60},
            {"yolo_landing_pad_class_id": 1},
            {"yolo_car_class_id": 0},

            # ----- Ground-plane projection -----
            {"yolo_ground_z": 1.5},
            {"yolo_ground_ray_z_epsilon": 1e-3},

            # ----- Camera -----
            {"camera_fov_horizontal": 2.7925268},
            {"camera_distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]},

            # ----- Buffers -----
            {"odometry_buffer_window_s": 1.0},
        ],
    )

    # Run independent gimbal controller
    gimbal_controller = Node(
        package="auto_lander",
        executable="gimbal_controller",
        name="gimbal_controller_node",
        output="screen",
        sigterm_timeout="20",
        sigkill_timeout="30",
        parameters=[
            {"use_sim_time": True},
            {"gimbal_control_rate": 20.0},
            {"gimbal_apriltag_target_max_age_s": 0.15},
            {"gimbal_yolo_target_max_age_s": 0.5},
            {"gimbal_kp": 0.03},
            {"gimbal_kd": 0.005},
            {"gimbal_max_slew_deg_s": 60.0},
            {"gimbal_initial_angle_deg": -90.0},
            {"gimbal_actual_pitch_initial_deg": -90.0},
            {"gimbal_servo_min_angle_deg": -90.0},
            {"gimbal_servo_max_angle_deg": 25.0},
            {"gimbal_yaw_command_deg": 0.0},
            {"camera_offset_x": 0.02},
            {"camera_offset_y": -0.01},
            {"camera_offset_z": -0.124923},
            {"camera_mount_roll": -1.5707963},
            {"camera_mount_pitch": 0.0},
            {"camera_mount_yaw": -1.5707963},
        ],
    )

    # Run main controller node
    # Run main controller node
    controller = Node(
        package="auto_lander",
        executable="controller",
        name="controller_node",
        output="screen",
        parameters=[
            # ----- General / safety -----
            {"diagnostics_enabled": True},
            {"ground_truth_available": True},
            {"use_sim_time": True},
            {"max_runtime": 200.0},
            {"boundary_limit": 600.0},

            # ----- Landing state / altitude parameters -----
            {"landing_recovery_height": 2.0},
            {"landing_height_above_gnd": 1.5},
            {"landing_height_threshold": 0.4},
            {"landing_error_threshold": 0.10},
            {"landing_centered_error_threshold": 2.0},

            {"landing_chase_altitude": 6.0},
            {"landing_descent_rate_far": -1.00},
            {"landing_descent_rate_close": -0.50},
            {"landing_time_visual_min_time (s)": 0.6},
            {"landing_pad_locked_time (s)": 10.0},
            {"landing_pad_lost_time (s)": 4.0},

            # ----- GPS loiter / waiting point -----
            {"gps_lat": -35.365876},
            {"gps_lon": 149.164137},
            {"gps_in_loc_buffer (m)": 2.0},

            # ----- Initial landing-pad state -----
            {"initial_position_state": [0.0, 0.0, 0.0]},
            {"initial_velocity_state": [0.0, 0.0, 0.0]},
            {"initial_yaw_state": 0.0},

            # ----- UKF parameters -----
            {"UKF_seeding_num_samples": 5},
            {"UKF_alpha": 1.0},
            {"UKF_beta": 2.0},
            {"UKF_kappa": 0.0},
            {"UKF_mahalanobis_threshold": 50.0},

            {"UKF_initial_P_diag": [
                1.00,
                1.00,
                1.00,
                1.00,
                2.00,
                0.20,
                0.20,
            ]},

            {"UKF_initial_Q_diag": [
                0.005,
                0.005,
                0.100,
                0.050,
                0.100,
                0.005,
                0.050,
            ]},

            {"UKF_unhealthy_covar": 1000.0},
            {"UKF_freq": 20.0},

            # ----- Controller loop -----
            {"CTRL_freq": 20.0},

            # ----- PID controller parameters -----
            {"PID_lam_0": 2.0},
            {"PID_Kp_0": 7.0},
            {"PID_Kd_0": 3.25},

            {"PID_Kp_pos_z": 0.2},
            {"PID_Kp_vel_z": 2.0},
            {"PID_Ki_vel_z": 0.2},
            {"PID_vel_z_i_clamp": 2.0},
            {"PID_pos_vel_err_limit": 1.5},

            {"PID_mass": 1.98},
            {"PID_max_thrust": 40.0},
            {"PID_gravity": 9.81},
            {"PID_drag_coefficient": 0.002},

            {"PID_max_throttle_rate": 1.0},
            {"PID_max_angle_rate": 1.0},

            {"PID_d_blend_start": 4.0},
            {"PID_d_blend_end": 1.0},
            {"PID_d_hold_radius": 4.0},
            {"PID_marker_yaw_sigma_threshold": 0.15},

            {"PID_drop_off_strength": 0.5},
            {"PID_terminal_gain": 1.0},
            {"PID_drop_off": 7.0},
            {"PID_accel_z_limit_g": 1.0},
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
            #apriltag_pose_detector,
            yolo_pose_detector,
            gimbal_controller,
            controller,
            delayed_agv_controller,
        ]
    )
