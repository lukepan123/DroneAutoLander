import os
from datetime import datetime

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue

""" REAL-FLIGHT launch (Pi + CUAV V6X / ArduCopter via MAVROS). MAVROS is started
    separately on the Pi's companion UART (FC SERIAL6):

    sudo systemctl stop fpv-switch      # it can grab /dev/ttyAMA0 and the gimbal
    ros2 run mavros mavros_node --ros-args -p fcu_url:=serial:///dev/ttyAMA0:921600 ...
    ros2 launch auto_lander main_launch.py gps_lat:=<deg> gps_lon:=<deg>

    Run both inside tmux on the Pi: if they live in an SSH session over the wfb-ng
    tunnel, a link drop kills the controller mid-flight.

    Flow: the controller does nothing until the PILOT selects GUIDED on the RC. It then
    arms, takes off, flies to the GPS waiting point and runs the landing state machine.
    Switching out of GUIDED at any time hands control back to the pilot for good.

    Everything marked TODO is a placeholder copied from SITL and MUST be measured /
    calibrated on the real hardware before flying.
"""

# ---- Camera: SIYI A2 mini, 1080p over RTSP, 160 deg horizontal (spec, same as SITL) ----
# Both must describe the frame AS PROCESSED (same 16:9 aspect as the stream). The A2
# applies its own distortion correction, so check how close to pinhole the output is.
CAMERA_HFOV_RAD = 2.7925268                      # TODO: calibrate (spec value, 160 deg)
CAMERA_DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]    # TODO: k1 k2 p1 p2 k3 from calibration

# ---- Camera mount in the drone body frame: TODO measure on the real airframe ----
# Read by the perception nodes (vision_common.load_camera_mount).
CAMERA_MOUNT = {
    "camera_offset_x": 0.02,
    "camera_offset_y": -0.01,
    "camera_offset_z": -0.124923,
    "camera_mount_roll": -1.5707963,
    "camera_mount_pitch": 0.0,
    "camera_mount_yaw": -1.5707963,
}

# ---- Test car + landing pad: TODO measure the real car ----
# Car frame: x forward, y left, z up, origin on the ground. 6 numbers per box:
# x_min, x_max, y_min, y_max, z_min, z_max (m). 

# Currently the SITL Prius.
# CAR_BOXES = [
#     -2.196,   2.126, -0.846,   0.846,  0.00,  0.85,    # lower body, bumpers, wheels
#     -2.139,   1.233, -0.846,   0.846,  0.85,  1.15,    # upper body, trunk deck
#     -1.332,   0.426, -0.608,   0.608,  1.15,  1.38,    # greenhouse
#     -1.094,   0.188, -0.608,   0.608,  1.38,  1.49,    # roof
#     -1.332,   0.378, -0.703,   0.703,  1.515, 1.525,   # landing platform
# ]
# PAD_HEIGHT_M = 1.525                             # pad top above the ground (m)
# CAR_PAD_POSITION = [-0.5, 0.0, PAD_HEIGHT_M]     # pad top-centre in the car frame
# CAR_REAR_AXLE = [-1.45, 0.0]                     # rear axle in the car frame (x, y)
# PAD_SIZE = [1.80, 1.48]                          # usable pad (length, width) (m)
# GEAR_SIZE = [0.3, 0.3]                           # landing-gear footprint (m)

# Grand Vitara 2008
CAR_BOXES = [
    -1.935,   2.235, -0.905,   0.905,  0.00,  1.00,    # body to beltline, bonnet, wheels
    -2.235,  -1.935, -0.360,   0.360,  0.50,  1.22,    # spare wheel on the tailgate
    -1.920,   0.700, -0.800,   0.800,  1.00,  1.45,    # greenhouse (windscreen rake averaged)
    -1.880,   0.300, -0.720,   0.720,  1.45,  1.695,   # roof skin + roof rails (spec height)
    -1.645,   0.065, -0.703,   0.703,  1.74,  1.75,    # landing platform (SITL pad size)
]
# Pad centred on the flat roof (x -1.88 .. +0.30). TODO measure: pad top height depends
# on the rails/crossbars and pad thickness, and is used directly as the landing height.
PAD_HEIGHT_M = 1.75                              # pad top above the ground (m), ESTIMATE
CAR_PAD_POSITION = [-0.79, 0.0, PAD_HEIGHT_M]    # pad top-centre in the car frame
CAR_REAR_AXLE = [-1.16, 0.0]                     # rear axle in the car frame (spec)
PAD_SIZE = [1.80, 1.48]                          # usable pad (length, width) (m), TODO real pad
GEAR_SIZE = [0.3, 0.3]                           # landing-gear footprint (m)



def generate_launch_description():
    launch_args = [
        # No defaults on purpose: the launch refuses to start without a real waiting point
        DeclareLaunchArgument("gps_lat", description="UAV GPS waiting point latitude (deg)"),
        DeclareLaunchArgument("gps_lon", description="UAV GPS waiting point longitude (deg)"),
        DeclareLaunchArgument("boundary_limit", default_value="80.0",
                              description="RTL if |x| or |y| from home exceeds this (m)"),
        DeclareLaunchArgument("max_runtime", default_value="180.0",
                              description="RTL this long after takeoff (s)"),
        # The A2 mini is an RTSP camera, not a V4L2 webcam: the AprilTag node's webcam
        # mode cannot open it. Something must publish /camera/image_raw from
        # rtsp://192.168.144.25:8554/main.264 (not yet in this repo).
        DeclareLaunchArgument("use_webcam", default_value="false",
                              description="true: the AprilTag node opens a USB/V4L2 webcam and "
                                          "republishes /camera/image_raw. false: an external "
                                          "camera driver publishes /camera/image_raw"),
        DeclareLaunchArgument("yolo_model", default_value="ugv_sim_yolo11n_openvino_model",
                              description="YOLO model path (absolute, or relative to the cwd)"),
        DeclareLaunchArgument("record_bag", default_value="true",
                              description="Record a rosbag of the key topics to ./bags/<run_id>"),
    ]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = os.path.join(os.getcwd(), "bags", run_id)

    # ----- Rosbag recorder (no images: too heavy for the Pi) -----
    bag_record = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration("record_bag")),
        cmd=[
            "ros2", "bag", "record",
            "-o", bag_dir,
            "/tf", "/tf_static",
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
            "/landing_pad/apriltag_target",
            "/landing_pad/gimbal_angle",
            "/landing_pad/rel_odom",
            "/landing_pad/pipeline_timing",
        ],
        output="screen",
        sigterm_timeout="20",   # give it time to finalise the bag on Ctrl+C
        sigkill_timeout="30",
    )

    # ----- AprilTag node: webcam owner (use_webcam:=true) -----
    # Opens the camera and republishes raw frames on /camera/image_raw for yolo.py.
    # Its tag detections drive the gimbal when visible; the UKF ignores them
    # (controller apriltag_enabled = False).
    apriltag_pose_detector = Node(
        condition=IfCondition(LaunchConfiguration("use_webcam")),
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
            {"image_source": "webcam"},
            {"use_sim_time": False},

            # ----- Webcam -----
            {"webcam_index": 0},
            {"webcam_fps": 30.0},

            # ----- Display / recording (headless Pi: no windows) -----
            {"show_debug_window": False},
            {"save_frames": False},
            {"create_video": False},           # YOLO node records the flight video
            {"video_fps": 10.0},
            {"output_dir": ""},

            # ----- Processing -----
            {"apriltag_processing_rate": 10.0},
            {"frame_capture_rate": 30.0},      # >= webcam_fps, so no stale frames queue up
            {"img_width": 960},                # 16:9, must match the capture aspect ratio
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

    # ----- YOLO car detector -----
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
            {"use_sim_time": False},

            # ----- Display / recording (headless Pi: no windows) -----
            {"show_debug_window": False},
            {"save_frames": False},
            {"create_video": True},
            {"video_fps": 10.0},
            {"output_dir": ""},

            # ----- Processing -----
            {"yolo_processing_rate": 10.0},
            {"img_width": 640},                # 16:9, same aspect as the AprilTag frame
            {"img_height": 360},

            # ----- YOLO -----
            {"yolo_enabled": True},
            {"yolo_model_path": ParameterValue(LaunchConfiguration("yolo_model"), value_type=str)},
            {"yolo_conf_threshold": 0.60},
            {"yolo_car_class_id": 0},

            # ----- Ground-ray check -----
            {"yolo_ground_ray_z_epsilon": 0.05},

            # ----- Camera -----
            {"camera_fov_horizontal": CAMERA_HFOV_RAD},
            {"camera_distortion_coeffs": CAMERA_DISTORTION},
            CAMERA_MOUNT,

            # ----- Buffers -----
            {"odometry_buffer_window_s": 1.0},
        ],
    )

    # ----- Independent gimbal controller -----
    gimbal_controller = Node(
        package="auto_lander",
        executable="gimbal_controller",
        name="gimbal_controller_node",
        output="screen",
        sigterm_timeout="20",
        sigkill_timeout="30",
        parameters=[
            {"use_sim_time": False},
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

    # ----- Main controller -----
    # Filter / guidance values mirror main_launch_sim.py (the SITL airframe is tuned to
    # match this quad), so real results stay comparable with SITL.
    controller = Node(
        package="auto_lander",
        executable="controller",
        name="controller_node",
        output="screen",
        sigterm_timeout="10",   # time to flush and close the CSV
        parameters=[
            # ----- General / safety -----
            {"diagnostics_enabled": True},
            {"ground_truth_available": False},    # no Gazebo truth topics in flight
            {"yolo_shadow_fit_enabled": False},   # SITL-only diagnostic, costs CPU
            {"use_sim_time": False},
            {"wait_for_pilot_guided": True},      # pilot flips to GUIDED to start
            {"mavros_message_rate_hz": 0.0},      # 0 = FCU SRx_* rates (see checklist)
            {"max_runtime": ParameterValue(LaunchConfiguration("max_runtime"), value_type=float)},
            {"boundary_limit": ParameterValue(LaunchConfiguration("boundary_limit"), value_type=float)},

            # ----- Landing state / altitude parameters -----
            {"landing_recovery_height": 7.0},
            {"landing_height_above_gnd": PAD_HEIGHT_M},
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
            {"landing_pad_size": PAD_SIZE},
            {"landing_gear_size": GEAR_SIZE},
            {"landing_car_a_max": 4.905},
            {"landing_car_a_lat_max": 5.0},
            {"landing_z_touchdown_accel_threshold": 1000.0},   # fallback disabled for first flights (SITL: 4.0)
            {"landing_z_touchdown_max_height": 1.5},   # no velocity-reversal cut above this

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
            {"UKF_buffer_window_s": 1.0},         # Pi YOLO latency > SITL; 0.5 s drops frames

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
            {"UKF_freq": 40.0},                   # drop to 20 if "UKF loop took" warnings appear
            {"apriltag_enabled": False},

            # ----- YOLO measurement model -----
            {"UKF_yolo_box_yaw_sigma_max": 0.25},
            {"UKF_yolo_box_pixel_sigma": 5.0},
            {"UKF_yolo_fit_yaw_enabled": True},
            {"yolo_roof_height_gain": 0.2},

            # ----- Car shape -----
            {"yolo_car_boxes": CAR_BOXES},
            {"yolo_car_pad_position": CAR_PAD_POSITION},
            {"yolo_car_rear_axle": CAR_REAR_AXLE},

            # ----- Controller loop -----
            {"CTRL_freq": 20.0},

            # ----- PID controller parameters -----
            {"PID_lam_0": 6.0},
            {"PID_Kp_0": 8.0},
            {"PID_Kd_0": 4.0},

            # Altitude: outer height loop only, FCU runs vel/accel/throttle
            {"PID_Kp_pos_z": 2.0},
            {"PID_vz_up_max": 2.0},           # MUST equal WPNAV_SPEED_UP / 100
            {"PID_vz_dn_max": 2.0},           # MUST equal WPNAV_SPEED_DN / 100
            {"PID_max_tilt": 0.70},           # rad (40 deg): ANGLE_MAX must be >= 4000

            {"PID_mass": 1.5},                # all-up weight (kg)
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

    return LaunchDescription(
        launch_args + [
            bag_record,
            apriltag_pose_detector,
            yolo_pose_detector,
            gimbal_controller,
            controller,
        ]
    )