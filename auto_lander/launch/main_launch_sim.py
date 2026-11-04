import os
from datetime import datetime

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue

""" SITL launch: Gazebo bridge, YOLO, gimbal, controller and the scripted AGV.

    e.g. ros2 launch auto_lander main_launch_sim.py trajectory:=sin speed_kmh:=40.0
"""

# ---- Camera (Gazebo iris_with_gimbal camera) ----
CAMERA_HFOV_RAD = 2.7925268                      # 160 deg, matches the SDF sensor
CAMERA_DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]    # ideal pinhole in sim

# Camera mount in the drone body frame. Read by the perception nodes
# (vision_common.load_camera_mount), not by the gimbal controller.
CAMERA_MOUNT = {
    "camera_offset_x": 0.02,
    "camera_offset_y": -0.01,
    "camera_offset_z": -0.124923,
    "camera_mount_roll": -1.5707963,
    "camera_mount_pitch": 0.0,
    "camera_mount_yaw": -1.5707963,
}


def generate_launch_description():
    # ----- Launch arguments (AGV test case; run_batch.sh sets these per run) -----
    launch_args = [
        DeclareLaunchArgument("trajectory", default_value="none",
                              description="none | straight | circle | sin | mbzirc | turn | mix1 | mix2"),
        DeclareLaunchArgument("speed_kmh", default_value="0.0",
                              description="AGV cruise speed (straight / circle / sin)"),
        DeclareLaunchArgument("accel_distance_m", default_value="50.0",
                              description="Distance over which the AGV reaches cruise speed"),
        DeclareLaunchArgument("turn_start_distance_m", default_value="20.0",
                              description="Straight cruise after reaching speed, before any turn"),
        DeclareLaunchArgument("sin_phase_deg", default_value="0.0",
                              description="Phase the sine weave starts at (deg); for mbzirc, "
                                          "the lap phase the car spawns at"),
        DeclareLaunchArgument("agv_add_noise", default_value="false",
                              description="Add Gaussian noise to the AGV v and w commands"),
        DeclareLaunchArgument("agv_delay_s", default_value="25.0",
                              description="Seconds from launch to AGV start (mbzirc batch runs use 5)"),
        # UAV GPS waiting point. Defaults = drone spawn in iris_runway_new.sdf;
        # mbzirc batch runs use the figure-8 centre (-35.365605, 149.164137)
        DeclareLaunchArgument("gps_lat", default_value="-35.365876",
                              description="UAV GPS waiting point latitude (deg)"),
        DeclareLaunchArgument("gps_lon", default_value="149.164137",
                              description="UAV GPS waiting point longitude (deg)"),
        DeclareLaunchArgument("record_bag", default_value="false",
                              description="Record a rosbag of the key topics to ./bags/<run_id>"),
    ]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = os.path.join(os.getcwd(), "bags", run_id)

    # ----- Rosbag recorder (record_bag:=true) -----
    bag_record = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration("record_bag")),
        cmd=[
            "ros2", "bag", "record",
            "--use-sim-time",
            "-o", bag_dir,
            "/tf", "/tf_static", "/clock",
            "/mavros/state",
            "/mavros/global_position/local",
            "/mavros/global_position/global",
            "/mavros/imu/data",
            "/mavros/setpoint_raw/attitude",
            "/mavros/setpoint_velocity/cmd_vel",
            "/mavros/setpoint_raw/global",
            "/mavros/gimbal_control/device/attitude_status",
            "/landing_pad/found",
            "/landing_pad/yolo_car_meas",
            "/landing_pad/yolo_target",
            "/landing_pad/gimbal_angle",
            "/landing_pad/rel_odom",
            "/landing_pad/pipeline_timing",
            "/quadcopter/true_odom",
            "/landing_pad/odom",
        ],
        output="screen",
        sigterm_timeout="20",   # give it time to finalise the bag on Ctrl+C
        sigkill_timeout="30",
    )

    # ----- Gazebo camera, odometry and AGV bridge -----
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

    # AprilTag landing pad detector (not launched in SITL; kept for reference)
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
            {"camera_fov_horizontal": CAMERA_HFOV_RAD},
            {"camera_distortion_coeffs": CAMERA_DISTORTION},
            CAMERA_MOUNT,

            # ----- Timestamp alignment buffers -----
            {"odometry_buffer_window_s": 1.0},
        ],
    )

    # YOLO car detector
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
            {"img_height": 360},

            # ----- YOLO -----
            {"yolo_enabled": True},
            {"yolo_model_path": "ugv_yolo11n_openvino_model"},
            {"yolo_conf_threshold": 0.50},
            {"yolo_car_class_id": 0},

            # ----- Ground-ray check -----
            # (car shape / pad height live on the controller, see yolo_car_boxes)
            {"yolo_ground_ray_z_epsilon": 0.05},

            # ----- Camera -----
            {"camera_fov_horizontal": CAMERA_HFOV_RAD},
            {"camera_distortion_coeffs": CAMERA_DISTORTION},
            CAMERA_MOUNT,

            # ----- Buffers -----
            {"odometry_buffer_window_s": 1.0},
        ],
    )

    # Independent gimbal controller
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
        ],
    )

    # Main controller
    controller = Node(
        package="auto_lander",
        executable="controller",
        name="controller_node",
        output="screen",
        parameters=[
            # ----- General / safety -----
            {"diagnostics_enabled": True},
            {"ground_truth_available": True},
            {"yolo_shadow_fit_enabled": True},    # SITL analysis; False on the Pi in flight
            {"use_sim_time": True},
            {"wait_for_pilot_guided": False},     # SITL: controller switches to GUIDED itself
            {"max_runtime": 200.0},
            {"boundary_limit": 800.0},

            # ----- Landing state / altitude parameters -----
            {"landing_recovery_height": 7.0},
            {"landing_height_above_gnd": 1.5},
            {"landing_height_threshold": 0.2},
            {"landing_centered_error_threshold": 3.0},

            {"landing_chase_altitude": 7.0},
            {"landing_descent_rate_far": -1.0},
            {"landing_descent_rate_close": -1.0},
            {"landing_time_visual_min_time (s)": 1.1},
            {"landing_pad_locked_time (s)": 30.0},
            {"landing_pad_lost_time (s)": 4.0},

            # ----- Throttle cut / termination check -----
            {"landing_cut_height_max": 1.0},
            {"landing_lead_time_max": 0.5},
            {"landing_cut_margin": 0.10},
            {"landing_pad_size": [1.80, 1.48]},   # sim platform box is 1.71 x 1.41 (0.95 scale)
            {"landing_gear_size": [0.3, 0.3]},
            {"landing_car_a_max": 4.905},
            {"landing_car_a_lat_max": 5.0},
            {"landing_z_touchdown_accel_threshold": 4.0},
            # landing_z_touchdown_max_height left at its default (inf = ungated)

            # ----- GPS loiter / waiting point -----
            {"gps_lat": ParameterValue(LaunchConfiguration("gps_lat"), value_type=float)},
            {"gps_lon": ParameterValue(LaunchConfiguration("gps_lon"), value_type=float)},
            {"gps_in_loc_buffer (m)": 2.0},

            # ----- Initial landing-pad state -----
            {"initial_position_state": [0.0, 0.0, 0.0]},
            {"initial_velocity_state": [0.0, 0.0, 0.0]},
            {"initial_yaw_state": 0.0},

            # ----- UKF parameters -----
            {"UKF_seeding_num_samples": 10},
            {"UKF_alpha": 0.5},
            {"UKF_beta": 2.0},
            {"UKF_kappa": 0.0},
            {"UKF_mahalanobis_threshold": 15.0},
            {"UKF_max_consecutive_rejections": 5},
            {"UKF_buffer_window_s": 0.5},

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
                0.001,
                0.001,
                0.010,
                0.030,
                0.050,
                0.001,
                0.050,
            ]},

            {"UKF_unhealthy_covar": 500.0},
            {"UKF_freq": 40.0},
            {"apriltag_enabled": False},

            # ----- YOLO measurement model -----
            {"UKF_yolo_box_yaw_sigma_max": 0.25},
            {"UKF_yolo_box_pixel_sigma": 5.0},
            {"UKF_yolo_fit_yaw_enabled": True},
            {"yolo_roof_height_gain": 0.2},

            # ----- Car shape (car model frame: x fwd, y left, z up; metres) -----
            # Frame = prius_hybrid base_link (same origin as /landing_pad/odom), origin at
            # ground level. Boxes fitted to the SDF collision primitives; mirrors estimated.
            # 6 numbers per box; add or remove rows freely.
            {"yolo_car_boxes": [
                # x_min   x_max   y_min    y_max   z_min  z_max     (xy scaled 0.95)
                -2.196,   2.126, -0.846,   0.846,  0.00,  0.85,    # lower body, bumpers, wheels
                -2.139,   1.233, -0.846,   0.846,  0.85,  1.15,    # upper body, trunk deck
                -1.332,   0.426, -0.608,   0.608,  1.15,  1.38,    # greenhouse
                -1.094,   0.188, -0.608,   0.608,  1.38,  1.49,    # roof
                -1.332,   0.378, -0.703,   0.703,  1.515, 1.525,   # landing platform
            ]},
            # Platform top-centre in the same frame. Keep plot_errors.py TRUE_OFFSET_* equal to it.
            {"yolo_car_pad_position": [-0.5, 0.0, 1.525]},
            {"yolo_car_rear_axle": [-1.45, 0.0]},

            # ----- Controller loop -----
            {"CTRL_freq": 20.0},

            # ----- PID controller parameters -----
            {"PID_lam_0": 6.0},
            {"PID_Kp_0": 8.0},
            {"PID_Kd_0": 4.0},

            # Altitude: outer height loop only, FCU runs vel/accel/throttle
            {"PID_Kp_pos_z": 2.0},
            {"PID_vz_up_max": 2.0},           # = WPNAV_SPEED_UP
            {"PID_vz_dn_max": 2.0},           # = WPNAV_SPEED_DN
            {"PID_max_tilt": 0.70},            # rad, <= ANGLE_MAX

            {"PID_mass": 1.48},
            {"PID_gravity": 9.81},
            {"PID_drag_coefficient": 0.03},

            {"PID_max_angle_rate": 1.0},

            {"PID_d_blend_start": 6.0},
            {"PID_d_blend_end": 3.0},
            {"PID_d_hold_radius": 6.0},
            {"PID_marker_yaw_sigma_threshold": 0.25},

            {"PID_drop_off_strength": 0.5},
            {"PID_terminal_gain": 1.0},
            {"PID_drop_off": 7.0},

            # Pad-acceleration feedforward (turns / speed changes). 0.0 = off (pure PD)
            {"PID_accel_ff_gain": 1.0},
        ],
    )

    # AGV controller (SITL only)
    agv_controller = Node(
        package="auto_lander",
        executable="agv_controller",
        name="agv_controller",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "trajectory": ParameterValue(LaunchConfiguration("trajectory"), value_type=str),
            "speed_kmh": ParameterValue(LaunchConfiguration("speed_kmh"), value_type=float),
            "accel_distance_m": ParameterValue(LaunchConfiguration("accel_distance_m"), value_type=float),
            "turn_start_distance_m": ParameterValue(LaunchConfiguration("turn_start_distance_m"), value_type=float),
            "add_noise": ParameterValue(LaunchConfiguration("agv_add_noise"), value_type=bool),

            # Turn shape: same steering profile at every speed (only v changes)
            "circle_yaw_rate": 0.2,     # rad/s, steady turn
            "sin_peak_yaw_rate": 0.2,   # rad/s, peak of the weave (matches the circle)
            "sin_period": 12.0,         # s
            "sin_phase_deg": ParameterValue(LaunchConfiguration("sin_phase_deg"), value_type=float),
        }],
    )
    delayed_agv_controller = TimerAction(
        period=LaunchConfiguration("agv_delay_s"),  # default 25 s; keep >= ~5 s so the gz bridge is up
        actions=[agv_controller],
    )

    return LaunchDescription(
        launch_args + [
            bag_record,
            gz_bridge,
            # apriltag_pose_detector,
            yolo_pose_detector,
            gimbal_controller,
            controller,
            delayed_agv_controller,
        ]
    )