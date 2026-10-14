import csv
import numpy as np
import tf_transformations
import tf2_ros
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy 
from rclpy.qos import HistoryPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import GlobalPositionTarget
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandLong
from mavros_msgs.srv import CommandTOL
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import MessageInterval
from datetime import datetime

from .state_definitions import QUAD_State
from .state_definitions import LP_State
from .state_definitions import LP_Meas
from .ukf import UKF
from .pid_controller import PIDController
from .pid_controller import CONTROL_OUTPUT_KEYS

""" Main orchestrator control loop, handles high level controller state logic and 
    orchestration of UKF and low-level controller for autonomous landing process. 

    Landing algorithim consists of several states.
"""

class Orchestrator(Node):
    """ Defines the orchestrator node."""

    def __init__(self) -> None:
        super().__init__("orchestrator")

        # region ---- STATE PARAMETERS ----
        # Enable logging and certain diagnostics
        self.declare_parameter("diagnostics_enabled", True)
        self.DIAGNOSTICS_ENABLED = self.get_parameter("diagnostics_enabled").get_parameter_value().bool_value
        # If in sim, log ground truth
        self.declare_parameter("ground_truth_available", True)
        self.GROUND_TRUTH_AVAILABLE = self.get_parameter("ground_truth_available").get_parameter_value().bool_value

        # Max runtime before RTL
        self.declare_parameter("max_runtime", 120.0)
        self.MAX_RUNTIME = self.get_parameter("max_runtime").get_parameter_value().double_value
        # Max boundary limit before RTL
        self.declare_parameter("boundary_limit", 30.0)
        self.BOUNDARY_LIMIT = self.get_parameter("boundary_limit").get_parameter_value().double_value

        # Landing height threshold params
        self.declare_parameter("landing_recovery_height", 2.0)
        self.LANDING_RECOVERY_HEIGHT = self.get_parameter("landing_recovery_height").get_parameter_value().double_value
        self.declare_parameter("landing_height_above_gnd", 1.5)
        self.LANDING_HEIGHT_ABOVE_GND = self.get_parameter("landing_height_above_gnd").get_parameter_value().double_value
        self.declare_parameter("landing_height_threshold", 0.20)
        self.LANDING_HEIGHT_THRESHOLD = self.get_parameter("landing_height_threshold").get_parameter_value().double_value
        # Touchdown fallback: detect a sharp downward-to-upward vertical velocity reversal.
        # This check is independent of the estimated height above the pad.
        self.declare_parameter("landing_z_touchdown_accel_threshold", 4.0)  # m/s^2
        self.LANDING_Z_TOUCHDOWN_ACCEL_THRESHOLD = self.get_parameter("landing_z_touchdown_accel_threshold").get_parameter_value().double_value
        self.declare_parameter("landing_centered_error_threshold", 2.0)
        self.LANDING_CENTERED_ERROR_THRESHOLD = self.get_parameter("landing_centered_error_threshold").get_parameter_value().double_value

        # Landing state timers/parameters
        self.declare_parameter("landing_chase_altitude", 6.0)
        self.LANDING_CHASE_HEIGHT_SP = self.get_parameter("landing_chase_altitude").get_parameter_value().double_value
        self.target_alt = self.LANDING_CHASE_HEIGHT_SP
        self.declare_parameter("landing_descent_rate_far", -1.00)
        self.DESCENT_RATE_FAR_SP = self.get_parameter("landing_descent_rate_far").get_parameter_value().double_value
        self.target_alt_rate = self.DESCENT_RATE_FAR_SP
        self.declare_parameter("landing_descent_rate_close", -0.50)
        self.DESCENT_RATE_CLOSE_SP = self.get_parameter("landing_descent_rate_close").get_parameter_value().double_value
        self.declare_parameter("landing_time_visual_min_time (s)", 0.6)
        self.LANDING_TIME_VISUAL_TIME_SP = self.get_parameter("landing_time_visual_min_time (s)").get_parameter_value().double_value
        self.declare_parameter("landing_pad_locked_time (s)", 10.0)
        self.LANDING_PAD_LOCK_TIME_SP = self.get_parameter("landing_pad_locked_time (s)").get_parameter_value().double_value
        self.declare_parameter("landing_pad_lost_time (s)", 4.0)
        self.LANDING_PAD_LOST_TIME_SP = self.get_parameter("landing_pad_lost_time (s)").get_parameter_value().double_value
        self.declare_parameter("landing_pad_found_timeout (s)", 0.5)
        self.LANDING_PAD_FOUND_TIMEOUT = self.get_parameter("landing_pad_found_timeout (s)").get_parameter_value().double_value

        # Loiter GPS Position
        self.declare_parameter("gps_lat", 0.00)
        self.GPS_LOITER_LAT = self.get_parameter("gps_lat").get_parameter_value().double_value
        self.declare_parameter("gps_lon", 0.00)
        self.GPS_LOITER_LON = self.get_parameter("gps_lon").get_parameter_value().double_value        
        self.gps_target = dict(lat=self.GPS_LOITER_LAT, lon=self.GPS_LOITER_LON, alt=self.target_alt + self.LANDING_HEIGHT_ABOVE_GND)
        self.declare_parameter("gps_in_loc_buffer (m)", 2.0)
        self.GPS_LOC_BUFFER = self.get_parameter("gps_in_loc_buffer (m)").get_parameter_value().double_value

        # Initial Landing Pad state
        self.declare_parameter('initial_position_state', [0.0, 0.0, 0.0])
        self.INIT_LP_POS = self.get_parameter("initial_position_state").get_parameter_value().double_array_value     
        self.landing_pad_relative_position = self.INIT_LP_POS
        self.landing_pad_relative_position_forward_predict = self.INIT_LP_POS
        self.declare_parameter('initial_velocity_state', [0.0, 0.0, 0.0])
        self.INIT_LP_VEL = self.get_parameter("initial_velocity_state").get_parameter_value().double_array_value     
        self.landing_pad_relative_velocity = self.INIT_LP_VEL
        self.landing_pad_relative_velocity_forward_predict = self.INIT_LP_VEL
        self.declare_parameter('initial_yaw_state', 0.0)
        self.INIT_LP_YAW = self.get_parameter("initial_yaw_state").get_parameter_value().double_value     
        self.landing_pad_yaw = self.INIT_LP_YAW
        self.landing_pad_yaw_forward_predict = self.INIT_LP_YAW

        # Dynamic throttle cut parameters
        self.declare_parameter("landing_lead_time_max", 0.5)
        self.LANDING_LEAD_TIME_MAX = self.get_parameter("landing_lead_time_max").get_parameter_value().double_value
        self.declare_parameter("landing_cut_height_max", 1.0)
        self.LANDING_CUT_HEIGHT_MAX = self.get_parameter("landing_cut_height_max").get_parameter_value().double_value
        self.declare_parameter("landing_cut_margin", 0.00)
        self.LANDING_CUT_MARGIN = self.get_parameter("landing_cut_margin").get_parameter_value().double_value
        self.declare_parameter("landing_pad_size", [1.80, 1.45])
        self.LANDING_PAD_SIZE = self.get_parameter("landing_pad_size").get_parameter_value().double_array_value
        self.declare_parameter("landing_gear_size", [0.3, 0.3])
        self.LANDING_GEAR_SIZE = self.get_parameter("landing_gear_size").get_parameter_value().double_array_value
        self.declare_parameter("landing_car_a_max", 4.905)
        self.LANDING_CAR_A_MAX = self.get_parameter("landing_car_a_max").get_parameter_value().double_value
        self.declare_parameter("landing_car_a_lat_max", 5.0)
        self.LANDING_CAR_A_LAT_MAX = self.get_parameter("landing_car_a_lat_max").get_parameter_value().double_value

        # ---- Global State Variables ----
        self.controller_state = 0
        self.fcu_state = State()
        self.odometry  = Odometry()
        self.quad_pose = np.zeros(3)
        self.quad_vel  = np.zeros(3)
        self.quad_z_accel = 0.0
        self._last_quad_z_velocity = None
        self._last_quad_z_velocity_time = None
        self._z_touchdown_trigger = False
        self._z_touchdown_trigger_accel = 0.0
        self._z_touchdown_trigger_prev_velocity = 0.0
        self._z_touchdown_trigger_velocity = 0.0
        self.quad_roll  = 0.0
        self.quad_pitch = 0.0
        self.quad_yaw   = 0.0
        self.landing_pad_relative_odometry = Odometry()
        self.alt_pos_control = True

        # ---- State 0xxx (Pre-arm) Variables ----
        self._mode_requested = False
        self._mode_confirmed = False
        self._mode = "GUIDED"
        self._arm_requested = False
        self._armed_confirmed = False
        self._armed_time = None

        # ---- State 1xxx (Take-off) Variables ----
        self._tko_requested = False
        self._tko_reached = False
        self._tko_complete_time = None
        self._tko_altitude_SP = self.target_alt + self.LANDING_HEIGHT_ABOVE_GND  # Inititalise at initial target altitude
        self._global_position = None

        # ---- State 2xxx (Searching for Landing Pad) Variables ----
        # "Found" = a detector reported the pad within this timeout (see _landing_pad_found)
        self._landing_pad_last_seen = None
        self._landing_pad_first_seen_time = None

        # ---- State 3xxx (Maintaining Landing Pad Lock) Variables ----
        self._landing_pad_locked_time_SP = self.LANDING_TIME_VISUAL_TIME_SP + self.LANDING_PAD_LOCK_TIME_SP
        self._landing_pad_lost_time = None

        # ---- State 4xxx (Beginning Landing Descent) Variables ----
        self._lead_time = 0.3
        self._landed_time = None
        self.cutoff = False

        # ---- State 5xxx (Landing Aborted) Variables ----
        self._landing_attempts = 0

        # ---- State 6xxx (Landing Confirmed) Variables ----
        self._idle_before_RTL_SP = 0.5
        self._disarm_requested = False
        self._disarmed = False

        # ---- State 7xxx (RTL) Variables ----
        self._rtl_initiated = False

        # region ---- UKF PARAMETERS ----
        self.declare_parameter('UKF_seeding_num_samples', 5)
        self.UKF_SEED_WINDOW_N = self.get_parameter("UKF_seeding_num_samples").get_parameter_value().integer_value
        self.declare_parameter('UKF_alpha', 1.0)
        self.UKF_ALPHA = self.get_parameter("UKF_alpha").get_parameter_value().double_value
        self.declare_parameter('UKF_beta', 2.0)
        self.UKF_BETA = self.get_parameter("UKF_beta").get_parameter_value().double_value
        self.declare_parameter('UKF_kappa', 0.0)
        self.UKF_KAPPA = self.get_parameter("UKF_kappa").get_parameter_value().double_value
        self.declare_parameter('UKF_mahalanobis_threshold', 50.0)
        self.UKF_MAHALANOBIS_THRESH = self.get_parameter("UKF_mahalanobis_threshold").get_parameter_value().double_value
        self.declare_parameter('UKF_initial_P_diag',
            [
                1.00,
                1.00,
                1.00,
                1.00,
                2.00,
                0.20,
                0.20,
            ])
        self.UKF_INIT_P_DIAG_ = self.get_parameter("UKF_initial_P_diag").get_parameter_value().double_array_value
        self.declare_parameter('UKF_initial_Q_diag', 
            [
                0.005,
                0.005,
                0.100,  # px, py, pz
                0.050,
                0.100,  # v, a
                0.005,
                0.050,  # yaw, yaw_rate
            ])
        self.UKF_INIT_Q_DIAG_ = self.get_parameter("UKF_initial_Q_diag").get_parameter_value().double_array_value
        self.declare_parameter('UKF_unhealthy_covar', 1000.0)
        self.UKF_UNHEALTHY_COVAR = self.get_parameter("UKF_unhealthy_covar").get_parameter_value().double_value
        self.declare_parameter('UKF_freq', 20.0)
        self.UKF_FREQ = self.get_parameter("UKF_freq").get_parameter_value().double_value  

        # AprilTag fusion toggle (the AprilTag node may not be running at all)
        self.declare_parameter('apriltag_enabled', False)
        self.APRILTAG_ENABLED = self.get_parameter("apriltag_enabled").get_parameter_value().bool_value

        # YOLO pixel-space box update: used once yaw is known well enough, otherwise the
        # yaw-agnostic box fit (position only) is fed instead, as during seeding
        self.declare_parameter('UKF_yolo_box_yaw_sigma_max', 0.25)   # rad
        self.UKF_YOLO_BOX_YAW_SIGMA_MAX = self.get_parameter("UKF_yolo_box_yaw_sigma_max").get_parameter_value().double_value
        self.declare_parameter('UKF_yolo_box_pixel_sigma', 25.0)      # px, per box edge
        self.UKF_YOLO_BOX_PIXEL_SIGMA = self.get_parameter("UKF_yolo_box_pixel_sigma").get_parameter_value().double_value

        # YOLO car model (moved from yolo.py)
        self.declare_parameter("yolo_car_roof_AGL", 1.5)
        self.YOLO_CAR_ROOF_AGL = self.get_parameter("yolo_car_roof_AGL").get_parameter_value().double_value
        self.declare_parameter("yolo_car_size_3d", [1.8, 4.3, 0.85])  # car extent (m): width, length, height (of body NOT car)
        self.YOLO_CAR_SIZE_3D = self.get_parameter("yolo_car_size_3d").get_parameter_value().double_array_value
        self.declare_parameter("yolo_car_fit_max_rel_rms", 0.08)  # max 3D-box fit residual, as a fraction of box size
        self.YOLO_CAR_FIT_MAX_REL_RMS = self.get_parameter("yolo_car_fit_max_rel_rms").get_parameter_value().double_value
        self.declare_parameter("yolo_roof_height_gain", 0.2)  # per car detection; how fast the roof height tracks (0 = fixed)
        self.YOLO_ROOF_HEIGHT_GAIN = self.get_parameter("yolo_roof_height_gain").get_parameter_value().double_value
        self.declare_parameter("yolo_car_roof_above_body", 0.65)  # roof height minus body-top height (m)
        self.YOLO_CAR_ROOF_ABOVE_BODY = self.get_parameter("yolo_car_roof_above_body").get_parameter_value().double_value

        # region ---- CTRL PARAMETERS ----
        self.declare_parameter('CTRL_freq', 20.0)
        self.CTRL_FREQ = self.get_parameter("CTRL_freq").get_parameter_value().double_value

        # Controller engineering / tuning parameters
        self.declare_parameter('PID_lam_0', 2.0)
        self.PID_LAM_0 = self.get_parameter("PID_lam_0").get_parameter_value().double_value
        self.declare_parameter('PID_Kp_0', 7.0)
        self.PID_KP_0 = self.get_parameter("PID_Kp_0").get_parameter_value().double_value
        self.declare_parameter('PID_Kd_0', 3.25)
        self.PID_KD_0 = self.get_parameter("PID_Kd_0").get_parameter_value().double_value

        self.declare_parameter('PID_Kp_pos_z', 1.0)
        self.PID_KP_POS_Z = self.get_parameter("PID_Kp_pos_z").get_parameter_value().double_value

        self.declare_parameter('PID_mass', 1.98)
        self.PID_MASS = self.get_parameter("PID_mass").get_parameter_value().double_value
        self.declare_parameter('PID_gravity', 9.81)
        self.PID_GRAVITY = self.get_parameter("PID_gravity").get_parameter_value().double_value
        self.declare_parameter('PID_drag_coefficient', 0.002)
        self.PID_DRAG_COEFFICIENT = self.get_parameter("PID_drag_coefficient").get_parameter_value().double_value

        self.declare_parameter('PID_vz_up_max', 2.0)
        self.PID_VZ_UP_MAX = self.get_parameter("PID_vz_up_max").get_parameter_value().double_value
        self.declare_parameter('PID_vz_dn_max', 2.0)
        self.PID_VZ_DN_MAX = self.get_parameter("PID_vz_dn_max").get_parameter_value().double_value
        self.declare_parameter('PID_max_tilt', 1.0)
        self.PID_MAX_TILT = self.get_parameter("PID_max_tilt").get_parameter_value().double_value
        self.declare_parameter('PID_max_angle_rate', 1.0)
        self.PID_MAX_ANGLE_RATE = self.get_parameter("PID_max_angle_rate").get_parameter_value().double_value

        self.declare_parameter('PID_d_blend_start', 4.0)
        self.PID_D_BLEND_START = self.get_parameter("PID_d_blend_start").get_parameter_value().double_value
        self.declare_parameter('PID_d_blend_end', 1.0)
        self.PID_D_BLEND_END = self.get_parameter("PID_d_blend_end").get_parameter_value().double_value
        self.declare_parameter('PID_d_hold_radius', 4.0)
        self.PID_D_HOLD_RADIUS = self.get_parameter("PID_d_hold_radius").get_parameter_value().double_value
        self.declare_parameter('PID_marker_yaw_sigma_threshold', 0.15)
        self.PID_MARKER_YAW_SIGMA_THRESHOLD = self.get_parameter("PID_marker_yaw_sigma_threshold").get_parameter_value().double_value
        self.declare_parameter('PID_marker_yaw_error_threshold', 0.3)
        self.PID_MARKER_YAW_ERROR_THRESHOLD = self.get_parameter("PID_marker_yaw_error_threshold").get_parameter_value().double_value

        self.declare_parameter('PID_drop_off_strength', 0.5)
        self.PID_DROP_OFF_STRENGTH = self.get_parameter("PID_drop_off_strength").get_parameter_value().double_value
        self.declare_parameter('PID_terminal_gain', 1.0)
        self.PID_TERMINAL_GAIN = self.get_parameter("PID_terminal_gain").get_parameter_value().double_value
        self.declare_parameter('PID_drop_off', 7.0)
        self.PID_DROP_OFF = self.get_parameter("PID_drop_off").get_parameter_value().double_value

        # region ---- INITIALISATIONS ----
        self._UKF_start = False
        self._UKF_seed_window_min_dt = self.LANDING_TIME_VISUAL_TIME_SP - 0.1

        self._UKF_forward_predict_x = None

        self._UKF_filter = UKF(
            self.UKF_INIT_P_DIAG_, self.UKF_INIT_Q_DIAG_,
            self.UKF_ALPHA, self.UKF_BETA, self.UKF_KAPPA,
            self.UKF_MAHALANOBIS_THRESH,
            self.INIT_LP_POS, self.INIT_LP_VEL, self.INIT_LP_YAW,
            self.UKF_SEED_WINDOW_N, self._UKF_seed_window_min_dt,
            self.YOLO_CAR_SIZE_3D, self.YOLO_CAR_ROOF_ABOVE_BODY, self.YOLO_CAR_ROOF_AGL,
            self.YOLO_CAR_FIT_MAX_REL_RMS, self.YOLO_ROOF_HEIGHT_GAIN,
        )

        self._UKF_diag = self._UKF_filter.get_covar_diagnostics()
        self._UKF_unhealthy_counter = 0

        self._pid_last_control_time = None
        self._pid_controller = PIDController(
            lam_0=self.PID_LAM_0,
            Kp_0=self.PID_KP_0,
            Kd_0=self.PID_KD_0,
            Kp_pos_z=self.PID_KP_POS_Z,
            m=self.PID_MASS,
            g=self.PID_GRAVITY,
            cD=self.PID_DRAG_COEFFICIENT,
            max_angle_rate=self.PID_MAX_ANGLE_RATE,
            d_blend_start=self.PID_D_BLEND_START,
            d_blend_end=self.PID_D_BLEND_END,
            d_hold_radius=self.PID_D_HOLD_RADIUS,
            drop_off_strength=self.PID_DROP_OFF_STRENGTH,
            terminal_gain=self.PID_TERMINAL_GAIN,
            drop_off=self.PID_DROP_OFF,
            vz_up_max=self.PID_VZ_UP_MAX,          # m/s, = WPNAV_SPEED_UP
            vz_dn_max=self.PID_VZ_DN_MAX,          # m/s, = WPNAV_SPEED_DN
            max_tilt=self.PID_MAX_TILT,            # rad, <= ANGLE_MAX
            platform_vz_ff=True # False unless the pad moves vertically
        )

        # region ---- SUBSCRIPTIONS ----
        _state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._state_sub = self.create_subscription(
            State, "/mavros/state", self._fcu_state_callback, _state_qos
        )

        _odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._odometry_sub = self.create_subscription(
            Odometry,
            "/mavros/global_position/local",
            self._odometry_callback,
            _odom_qos,
        )

        self._gps_sub = self.create_subscription(
            NavSatFix,
            "/mavros/global_position/global",
            self._gps_callback,
            qos_profile_sensor_data,
        )

        self._landing_pad_found_sub = self.create_subscription(
            Bool, "/landing_pad/found", self._landing_pad_found_callback, 1
        )

        # Raw YOLO car box + image-time camera geometry (layout documented in yolo.py)
        self._yolo_msg = None
        self._yolo_meas_sub = self.create_subscription(
            Float64MultiArray, "/landing_pad/yolo_car_meas", self._yolo_measurement_callback, 10
        )

        # region ---- PUBLISHERS ----
        self.att_pub = self.create_publisher(
            AttitudeTarget, "/mavros/setpoint_raw/attitude", 10
        )
        self.vel_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10
        )
        self.global_pos_pub = self.create_publisher(
            GlobalPositionTarget, "/mavros/setpoint_raw/global", 10
        )

        # region ---- TF2 ----
        self._tf_map_landing_pad_buffer = tf2_ros.Buffer(
            node=self, cache_time=Duration(seconds=10)
        )
        self._tf_map_landing_pad_listener = tf2_ros.TransformListener(
            self._tf_map_landing_pad_buffer, self
        )

        # region ---- CLIENTS ----
        self._set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self._arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self._takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self._command_long_client = self.create_client(CommandLong, "/mavros/cmd/command")
        self._message_interval_client = self.create_client(
            MessageInterval, "/mavros/set_message_interval"
        )

        # region ---- CONTROL TIMERS ----
        self._UKF_timer = self.create_timer(1.0/self.UKF_FREQ, self._UKF_loop)
        self._safety_timer_rate = 1.0
        self._safety_timer = self.create_timer(
            self._safety_timer_rate, self._safety_loop
        )
        self._control_timer = self.create_timer(
            1.0/self.CTRL_FREQ, self._control_loop
        )

        self._set_message_intervals()  # MAVROS message rates

        # region ---- DIAGNOSTICS AND LOGGING ----
        """ Generate a log if diagnostics and logging enabled."""
        self.quad_true_odometry = Odometry()
        self._quad_true_recv = 0.0
        self._pad_true_recv = 0.0
        self._UKF_est_time = 0.0
        self._UKF_fwd_horizon = 0.0
        self._rtl_reason = ""
        self.landing_pad_true_odometry = Odometry()
        if self.DIAGNOSTICS_ENABLED:
            self.start_diagnostics()
        
        self.get_logger().info("Auto Lander (callbacks) started")


    # region ---- CALLBACK IMPLEMENTATIONS ----
    def _set_message_intervals(self) -> None:
        """ Set specfied message interval rates.
        """
        # rate = 40.0  # Hz
        # # Drives global_position/local odometry
        # self._set_single_message_interval(33, rate, "GLOBAL_POSITION_INT")
        # # Drives local_position/velocity_local
        # self._set_single_message_interval(32, rate, "LOCAL_POSITION_NED")
        # # Orientation/angular-rate freshness for both above
        # self._set_single_message_interval(31, rate, "ATTITUDE_QUATERNION")


    def _set_single_message_interval(self, message_id, rate, description) -> None:
        """ Set a single message interval.
        
        :param message_id: ID of the message whose interval is being set
        :param rate: Rate (Hz) to set message ID to
        :param description: Description of message ID
        """
        if not self._message_interval_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f"MessageInterval service not ready for {description}"
            )
            return

        req = MessageInterval.Request()
        req.message_id = message_id
        req.message_rate = rate

        fut = self._message_interval_client.call_async(req)
        fut.add_done_callback(
            lambda f, desc=description, mid=message_id, r=rate: self._on_message_interval_done(
                f, desc, mid, r
            )
        )

        self.get_logger().info(f"Setting {description} (ID: {message_id}) to {rate}Hz")


    def _on_message_interval_done(self, fut, description, message_id, rate) -> None:
        """ Handle message interval service response.
        
        :param fut: Client object
        :param description: Description of message ID
        :param message_id: ID of the message whose interval is being set
        :param rate: Rate (Hz) to set message ID to
        """
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"{description} interval setting exception: {e}")
            return

        if getattr(res, "success", False):
            self.get_logger().info(
                f"{description} interval set to {rate}Hz successfully"
            )
        else:
            self.get_logger().warn(
                f"{description} interval setting failed (ID: {message_id}, Rate: {rate}Hz)"
            )


    def _fcu_state_callback(self, msg: State) -> None:
        """ Confirm state from FCU, and if good arm the FCU.
        
        :param msg: State msg from FCU
        """

        self.fcu_state = msg

        if self.fcu_state.mode == self._mode and not self._mode_confirmed:
            self._mode_confirmed = True
            self.get_logger().info(f"{self._mode} confirmed by FCU.")

        if self.fcu_state.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = self.get_clock().now().nanoseconds / 1e9

        if self._armed_confirmed and not self.fcu_state.armed:
            self._disarmed = True


    def _odometry_callback(self, msg: Odometry) -> None:
        """ Obtain FCU Odometry, process quaternions into euler roll, pitch and yaw 
            values. Also monitors takeoff condition.

        :param msg: Odometry msg from quadcopter
        """

        self.odometry = msg
        alt = msg.pose.pose.position.z

        q = self.odometry.pose.pose.orientation
        self.quad_roll, self.quad_pitch, self.quad_yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        self.quad_pose = np.array(
            [
                self.odometry.pose.pose.position.x,
                self.odometry.pose.pose.position.y,
                self.odometry.pose.pose.position.z,
            ]
        )

        # For some reason velocity is negative (invert so that up = +, down= -)
        self.quad_vel = np.array(
            [
                self.odometry.twist.twist.linear.x,
                self.odometry.twist.twist.linear.y,
                -self.odometry.twist.twist.linear.z,
            ]
        )

        # Estimate vertical acceleration from successive odometry velocities.
        # The sign convention above is up = positive, down = negative.
        z_velocity = float(self.quad_vel[QUAD_State.Z])
        z_velocity_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if z_velocity_time <= 0.0:
            z_velocity_time = self.get_clock().now().nanoseconds * 1e-9

        if self._last_quad_z_velocity_time is not None:
            z_dt = z_velocity_time - self._last_quad_z_velocity_time
            if 0.005 <= z_dt <= 0.25:
                self.quad_z_accel = (z_velocity - self._last_quad_z_velocity) / z_dt # type: ignore

                # During state 4000, a sharp down-to-up velocity reversal can indicate
                # impact/rebound. Do not gate this fallback on the estimated pad height.
                velocity_rising = (self._last_quad_z_velocity < z_velocity) #type: ignore

                if (
                    self.controller_state == 4000
                    and velocity_rising
                    and self.quad_z_accel >= self.LANDING_Z_TOUCHDOWN_ACCEL_THRESHOLD
                ):
                    self._z_touchdown_trigger = True
                    self._z_touchdown_trigger_accel = self.quad_z_accel
                    self._z_touchdown_trigger_prev_velocity = self._last_quad_z_velocity
                    self._z_touchdown_trigger_velocity = z_velocity

        self._last_quad_z_velocity = z_velocity
        self._last_quad_z_velocity_time = z_velocity_time

        if (
            self._armed_confirmed
            and not self._tko_reached
            and alt > (self._tko_altitude_SP - 0.5)
        ):
            self._tko_reached = True
            self._tko_complete_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(
                f"Takeoff complete at {alt:.2f} m - Starting {self.MAX_RUNTIME}s safety timer"
            )


    def _gps_callback(self, msg) -> None:
        """Obtain GPS coords.
        
        :param msg: GPS coords
        """

        self._global_position = msg


    def _landing_pad_found_callback(self, msg: Bool) -> None:
        """ Record when a detector last saw the pad. False messages are ignored: YOLO
            only ever publishes True, so "found" has to decay with time rather than
            wait for a False that may never come (e.g. AprilTag node off).

        :param msg: Bool msg from the AprilTag / YOLO nodes
        """

        if msg.data:
            self._landing_pad_last_seen = self.get_clock().now().nanoseconds * 1e-9


    @property
    def _landing_pad_found(self) -> bool:
        """ True while a detection arrived within LANDING_PAD_FOUND_TIMEOUT. """
        return (
            self._landing_pad_last_seen is not None
            and self.get_clock().now().nanoseconds * 1e-9 - self._landing_pad_last_seen
            < self.LANDING_PAD_FOUND_TIMEOUT
        )


    def _yolo_measurement_callback(self, msg: Float64MultiArray) -> None:
        """ Store the latest raw YOLO car measurement; consumed once by the UKF loop.

        :param msg: Raw box + camera geometry from yolo.py
        """

        self._yolo_msg = msg


    def _request_mode(self) -> None:
        """Set mode on FCU.
        """

        if not self._set_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SetMode service not ready yet.")
            return
        self._mode_requested = True
        req = SetMode.Request()
        req.custom_mode = self._mode
        fut = self._set_mode_client.call_async(req)
        fut.add_done_callback(self._on_set_mode_done)
        self.get_logger().info(f"Requesting {self._mode}...")

    def _on_set_mode_done(self, fut) -> None:
        """Check if mode properly set on FCU.
        
        :param fut: Client object
        """

        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"SetMode exception: {e}")
            self._mode_requested = False
            return

        if getattr(res, "mode_sent", False):
            self.get_logger().info(
                f"{self._mode} command accepted (awaiting FCU report)."
            )
        else:
            self.get_logger().error(f"{self._mode} command rejected by FCU.")
            self._mode_requested = False


    def _request_arm(self) -> None:
        """Request FCU Arm.
        """

        if not self._arming_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Arming service not ready yet.")
            return
        self._arm_requested = True
        req = CommandBool.Request()
        req.value = True
        fut = self._arming_client.call_async(req)
        fut.add_done_callback(self._on_arm_done)
        self.get_logger().info("Requesting ARM...")


    def _on_arm_done(self, fut) -> None:
        """Check if FCU Armed.
        
        :param fut: Client object
        """

        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"Arming exception: {e}")
            self._arm_requested = False
            return

        if getattr(res, "success", False):
            self.get_logger().info("Arm accepted (awaiting FCU armed=true).")
        else:
            self.get_logger().error(
                f'Arm rejected by FCU (result={getattr(res, "result", None)}).'
            )
            self._arm_requested = False


    def _request_takeoff(self) -> None:
        """Request FCU takeoff.
        """

        if not self._takeoff_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Takeoff service not ready yet.")
            return
        self._tko_requested = True
        req = CommandTOL.Request()
        req.altitude = float(self._tko_altitude_SP)
        fut = self._takeoff_client.call_async(req)
        fut.add_done_callback(self._on_takeoff_done)
        self.get_logger().info(
            f"Requesting takeoff to {self._tko_altitude_SP:.1f} m..."
        )

    def _on_takeoff_done(self, fut):
        """Confirm FCU takeoff successful.
        
        :param fut: Client object
        """

        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"Takeoff exception: {e}")
            self._tko_requested = False
            return

        if getattr(res, "success", False):
            self.get_logger().info("Takeoff command accepted (monitoring altitude).")
        else:
            self.get_logger().error("Takeoff rejected by FCU.")
            self._tko_requested = False


    def _initiate_rtl(self, reason) -> None:
        """Initiate return to land mode and disable tracking.
        
        :param reason: Reason for RTL
        """

        if self._rtl_initiated:
            return

        self._rtl_initiated = True
        self._rtl_reason = str(reason)

        self.get_logger().warn(f"SAFETY: Initiating RTL due to {reason}")

        if not self._set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("SetMode service not available for RTL!")
            return

        req = SetMode.Request()
        req.custom_mode = "RTL"
        fut = self._set_mode_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_rtl_done(f, reason))


    def _on_rtl_done(self, fut, reason):
        """Handle RTL mode change response.
        
        :param fut: Client object
        :param reason: Reason for RTL
        """

        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"RTL exception: {e}")
            return

        if getattr(res, "mode_sent", False):
            self.get_logger().info(f"RTL command accepted (reason: {reason})")
        else:
            self.get_logger().error(f"RTL command rejected by FCU (reason: {reason})")


    def _request_disarm(self) -> None:
        """Force-disarm the FCU once via MAV_CMD_COMPONENT_ARM_DISARM (param2=21196
        bypasses the FCU's landed/safety checks). If it refuses, we leave it."""
        if not self._command_long_client.service_is_ready():
            self.get_logger().warn("CommandLong service not ready for force disarm.", throttle_duration_sec=2.0)
            return
        self._disarm_requested = True
        req = CommandLong.Request()
        req.broadcast = False
        req.command = 400          # MAV_CMD_COMPONENT_ARM_DISARM
        req.confirmation = 0
        req.param1 = 0.0           # 0 = disarm
        req.param2 = 21196.0       # force-disarm magic number
        req.param3 = 0.0
        req.param4 = 0.0
        req.param5 = 0.0
        req.param6 = 0.0
        req.param7 = 0.0
        fut = self._command_long_client.call_async(req)
        fut.add_done_callback(self._on_disarm_done)
        self.get_logger().info("Requesting FORCE DISARM...")


    def _on_disarm_done(self, fut) -> None:
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"Force disarm exception: {e}")
            return
        if getattr(res, "success", False):
            self.get_logger().info("Force disarm accepted.")
        else:
            self.get_logger().warn(f'Force disarm rejected by FCU (result={getattr(res, "result", None)}).')


    # region ---- UKF LOOP ----
    def _UKF_loop(self):
        """ Run UKF predict and update loops. Manages state estimation of the landing
            pad platform. Calls upon the YOLO car measurement and, if enabled, the 
            AprilTag measurement.
        """

        # Latch start of UKF for landing pad on acquisition of landing pad from detector
        # node. The control loop clears _UKF_start when the pad is declared lost, so a
        # reacquisition always resets and re-seeds the filter.
        if self._landing_pad_found and not self._UKF_start and self.controller_state >= 2000:
            self._UKF_filter.reset()
            # Initalise to something reasonable
            self._UKF_filter.x[LP_State.PZ] = self.LANDING_HEIGHT_ABOVE_GND - self.quad_pose[QUAD_State.Z]
            self._UKF_start = True

        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9

        # Predict UKF step up to now (after seeding). The filter steps by the gap since
        # its own last event, so a seed/measurement stamped in the past is caught up.
        if self._UKF_start and not self._UKF_filter.seed_pending:
            self._UKF_filter.predict(self.quad_vel, now_sec)

        # Warn in logs if filter goes non-PD or any sigma blows up
        if not self._UKF_diag["is_pd"]:
            self.get_logger().warn(
                "UKF: covariance matrix is no longer positive definite!"
            )

        # Update measurement noise based on drone angular rates
        cov_adj_x = 1 + 2 * np.sqrt(
            self.odometry.twist.twist.angular.y**2
            + self.odometry.twist.twist.angular.z**2
        )
        cov_adj_y = 1 + 2 * np.sqrt(
            self.odometry.twist.twist.angular.x**2
            + self.odometry.twist.twist.angular.z**2
        )
        cov_adj_z = 1 + 2 * np.sqrt(
            self.odometry.twist.twist.angular.x**2
            + self.odometry.twist.twist.angular.y**2
        )

        R_apriltag = np.diag([
            0.10 * cov_adj_x,
            0.10 * cov_adj_y,
            0.10 * cov_adj_z,
            0.10 * cov_adj_z,
        ])

        # Update YOLO measurement noise, inflate covariance as altitude drops since
        # the bounding box becomes a worse estimate (clipping, car != box at close range).
        alt = -self.landing_pad_relative_position_forward_predict[LP_State.PZ]

        d_near, d_far = 0.0, self.LANDING_HEIGHT_ABOVE_GND + 2.0
        scale_max, gamma = 10.0, 2.0
        scale_t = np.clip((d_far - alt) / (d_far - d_near), 0.0, 1.0)
        yolo_car_scale = 1.0 + (scale_max - 1.0) * scale_t ** gamma

        # Fallback (fitted position) noise. Trust the fitted z only when the error
        # between the quad yaw and LP yaw is within a certain threshold, and we are not
        # too uncertain about the current LP yaw (the fit assumes the car is roughly
        # aligned with the camera heading). With the default thresholds the fallback only
        # runs when sigma_yaw >= UKF_yolo_box_yaw_sigma_max (> the 0.15 here), so this
        # only matters for the fit-only baseline (UKF_yolo_box_yaw_sigma_max = 0.0).
        yaw_error = abs(self._UKF_filter._wrap(self.quad_yaw - self.landing_pad_yaw)) #type: ignore
        if (
            self._UKF_diag["sigma_yaw"] < self.PID_MARKER_YAW_SIGMA_THRESHOLD
            and yaw_error < self.PID_MARKER_YAW_ERROR_THRESHOLD
        ):
            car_z_var = 0.25
        else:
            car_z_var = 5.0

        R_yolo_car = np.diag([
            0.500 * yolo_car_scale,
            0.500 * yolo_car_scale,
            car_z_var,
            1e9,
        ])

        # Pixel-space box noise, with the same close-range inflation
        R_yolo_box = np.eye(4) * self.UKF_YOLO_BOX_PIXEL_SIGMA**2 * yolo_car_scale

        # Gather measurements as (type, stamp, z, R, cam, pipeline_timing)
        measurements = []

        # AprilTag (optional): full pose from TF
        if self.APRILTAG_ENABLED:
            try:
                tf_msg = self._tf_map_landing_pad_buffer.lookup_transform(
                    "local", "landing_pad_link", Time()
                )
                t, q = tf_msg.transform.translation, tf_msg.transform.rotation
                _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
                key = (tf_msg.header.stamp.sec, tf_msg.header.stamp.nanosec)
                measurements.append((
                    "apriltag", key[0] + key[1] * 1e-9, np.array([t.x, t.y, t.z, yaw]),
                    R_apriltag, None,
                    self._pipeline_timing_buffer.pop(key, None) if self.DIAGNOSTICS_ENABLED else None,
                ))
            except tf2_ros.TransformException: #type: ignore
                pass

        # YOLO car box: consumed once, so the fallback fit's roof learning runs once per frame
        msg, self._yolo_msg = self._yolo_msg, None
        if msg is not None:
            d = np.asarray(msg.data, dtype=float)
            key = (int(d[0]), int(d[1]))
            stamp = key[0] + key[1] * 1e-9
            K = np.array([[d[8], 0.0, d[10]], [0.0, d[9], d[11]], [0.0, 0.0, 1.0]])
            cam = (d[15:24].reshape(3, 3), d[12:15], K, d[7])
            timing = self._yolo_pipeline_timing_buffer.pop(key, None) if self.DIAGNOSTICS_ENABLED else None

            sigma_yaw = np.sqrt(max(self._UKF_filter.P[LP_State.YAW, LP_State.YAW], 0.0))
            if self._UKF_filter.seed_pending or sigma_yaw >= self.UKF_YOLO_BOX_YAW_SIGMA_MAX:
                # Yaw too poor for the pixel model: yaw-agnostic box fit -> position only
                pos = self._UKF_filter.fit_car_box(d[2:6], cam)
                if pos is not None:
                    measurements.append(("yolo", stamp, np.append(pos, 0.0), R_yolo_car, None, timing))
            else:
                measurements.append(("yolo", stamp, d[2:6], R_yolo_box, cam, timing))

        # Update each measurement stream through the same UKF path.
        for measurement_type, stamp, z, R, cam, timing in measurements:
            if not self._UKF_filter.update(measurement_type, stamp, z, R, now_sec, cam, timing):
                self.get_logger().warn(
                    f"UKF {measurement_type} update failed!",
                    throttle_duration_sec=2.0,
                )

        # ---- Grab per-state covariance diagnostics every tick ----
        self._UKF_diag = self._UKF_filter.get_covar_diagnostics()

        # Always publish current UKF state
        x = self._UKF_filter.x

        # Forward predict by the time it would take for the drone to drop from its
        # altitude to the landing pad. This time is derived from the termination state
        # Time to landing assuming 1g acceleration to max descent rate, then constant max rate.
        h = max(-x[LP_State.PZ], 0.0)
        v = max(-self.quad_vel[QUAD_State.Z], 0.0)
        v_max = abs(self.PID_VZ_DN_MAX)
        a = 9.81

        t_acc = max(v_max - v, 0.0) / a
        d_acc = v * t_acc + 0.5 * a * t_acc**2

        if d_acc >= h:
            fall_t = (-v + np.sqrt(v**2 + 2*a*h)) / a
        else:
            fall_t = t_acc + (h - d_acc) / v_max

        self._lead_time = min(fall_t, self.LANDING_LEAD_TIME_MAX)
        t = self._lead_time

        # Also compensate for the drag on the drone
        quad_vel_eff = np.array(self.quad_vel, dtype=float)
        k = self.PID_DRAG_COEFFICIENT * np.hypot(quad_vel_eff[QUAD_State.X], quad_vel_eff[QUAD_State.Y]) * t
        if k < 1e-6:
            scale = 1.0
        else:
            scale =  np.log1p(k) / k

        quad_vel_eff[QUAD_State.X] *= scale
        quad_vel_eff[QUAD_State.Y] *= scale

        self._UKF_est_time = now_sec   # instant the estimate is valid at
        self._UKF_fwd_horizon = t
        self._UKF_forward_predict_x = self._UKF_filter.forward_predict(quad_vel_eff, t)

        self.landing_pad_relative_position_forward_predict = np.array(
            [self._UKF_forward_predict_x[LP_State.PX],
             self._UKF_forward_predict_x[LP_State.PY],
             self._UKF_forward_predict_x[LP_State.PZ]]
        )

        # Relative velocity = pad velocity (CTRA: speed along heading, no vertical) - quad
        fwd_yaw = self._UKF_forward_predict_x[LP_State.YAW]
        self.landing_pad_yaw_forward_predict = fwd_yaw
        self.landing_pad_relative_velocity_forward_predict = np.array([
            self._UKF_forward_predict_x[LP_State.V] * np.cos(fwd_yaw) - self.quad_vel[0],
            self._UKF_forward_predict_x[LP_State.V] * np.sin(fwd_yaw) - self.quad_vel[1],
            -self.quad_vel[2],
        ])

        # Publish the actual landing pad pose estimate for all other uses
        # (diagnostics etc.)
        self.landing_pad_relative_position = np.array(
            [x[LP_State.PX], x[LP_State.PY], x[LP_State.PZ]]
        )

        yaw = x[LP_State.YAW]
        rel_vx = x[LP_State.V] * np.cos(yaw) - self.quad_vel[0]
        rel_vy = x[LP_State.V] * np.sin(yaw) - self.quad_vel[1]
        rel_vz = -self.quad_vel[2]
        self.landing_pad_relative_velocity = np.array([rel_vx, rel_vy, rel_vz])

        self.landing_pad_relative_odometry.header.stamp = (
            self.get_clock().now().to_msg()
        )
        self.landing_pad_relative_odometry.header.frame_id = "local"

        self.landing_pad_relative_odometry.pose.pose.position.x = x[LP_State.PX]
        self.landing_pad_relative_odometry.pose.pose.position.y = x[LP_State.PY]
        self.landing_pad_relative_odometry.pose.pose.position.z = x[LP_State.PZ]

        quat = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
        self.landing_pad_relative_odometry.pose.pose.orientation.x = quat[0]
        self.landing_pad_relative_odometry.pose.pose.orientation.y = quat[1]
        self.landing_pad_relative_odometry.pose.pose.orientation.z = quat[2]
        self.landing_pad_relative_odometry.pose.pose.orientation.w = quat[3]
        self.landing_pad_yaw = yaw

        self.landing_pad_relative_odometry.twist.twist.linear.x = float(rel_vx)
        self.landing_pad_relative_odometry.twist.twist.linear.y = float(rel_vy)
        self.landing_pad_relative_odometry.twist.twist.linear.z = float(rel_vz)

        # Check for timer overruns
        end = self.get_clock().now()
        elapsed = (end - now).nanoseconds / 1e6
        if elapsed > 1.0/(self.UKF_FREQ) * 1000:
            self.get_logger().warn(f"UKF loop took {elapsed:.2f} ms!")


    # region ---- CTRL LOOP ----
    def _control_loop(self) -> None:
        """ Run main control and orchestration loop. Handles state machine logic.
        """

        # Get timestamp
        start = self.get_clock().now()
        t_now = start.nanoseconds * 1e-9

        # Reset logging
        self._pid_controller.reset_outputs()

        # Horizontal error to the forward-predicted pad
        err = np.hypot(*self.landing_pad_relative_position_forward_predict[:2]) #type: ignore

        # ---- State 0000 (Pre-arm)
        # Below conditions need to be met ALWAYS so we check regardless of state
        if not self.fcu_state.connected:
            self.controller_state = 0
            return

        if not self._mode_confirmed:
            if not self._mode_requested:
                self._request_mode()
            self.controller_state = 0
            return

        if not self._armed_confirmed:
            if not self._arm_requested:
                self._request_arm()
            self.controller_state = 0
            return

        # If RTL initiated, exit early
        if self._rtl_initiated:
            self.controller_state = 7100
            if self.DIAGNOSTICS_ENABLED:
                self.log_diagnostics()
            return

        if (
            self._armed_confirmed
            and self._armed_time is not None
            and not self._tko_requested
        ):
            self.controller_state = 1000

        # ---- State 1000 (Start Takeoff)
        if self.controller_state == 1000:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._armed_time
            if elapsed < 5.0:
                if (
                    not hasattr(self, "_armed_wait_logged")
                    or not self._armed_wait_logged
                ):
                    self.get_logger().info("Armed. Waiting 5s before takeoff...")
                    self._armed_wait_logged = True
                return
            else:
                self._request_takeoff()
                self._armed_wait_logged = False
                self.controller_state = 1100
                return

        # ---- State 1100 (Wait for Takeoff to finish)
        if self.controller_state == 1100 and self._tko_reached:
            self.controller_state = 1200

        # ---- State 1200 (Go to predesignated waiting point)
        if self.controller_state == 1200:
            if self._global_position is not None:
                d_north = (self.gps_target["lat"] - self._global_position.latitude) * 111132.0
                d_east = (self.gps_target["lon"] - self._global_position.longitude) * 111320.0 * np.cos(np.radians(self.gps_target["lat"]))

                if np.hypot(d_north, d_east) <= self.GPS_LOC_BUFFER:
                    self.get_logger().info(f"At designated wait position...")
                    self.controller_state = 2000

        # Discard any stale touchdown event if we are not in the landing-descent state.
        if self.controller_state != 4000:
            self._z_touchdown_trigger = False

        # ---- State 2000 (Searching for Landing Pad)
        if self.controller_state == 2000:
            self.target_alt = self.LANDING_CHASE_HEIGHT_SP
            if self._landing_pad_found and self._landing_pad_first_seen_time is None:
                self._landing_pad_first_seen_time = t_now
                self.get_logger().info(
                    f"Landing Pad found, starting {self.LANDING_TIME_VISUAL_TIME_SP} sec timer..."
                )
                self.controller_state = 2100

        # ---- Pad-lost watchdog (states 2100-5999) ----
        # One timer for every tracking state: lost for longer than
        # LANDING_PAD_LOST_TIME_SP -> back to searching, and drop the UKF so the next
        # acquisition resets and re-seeds it.
        if 2100 <= self.controller_state < 6000:
            if self._landing_pad_found:
                self._landing_pad_lost_time = None
            elif self._landing_pad_lost_time is None:
                self._landing_pad_lost_time = t_now
            elif t_now - self._landing_pad_lost_time > self.LANDING_PAD_LOST_TIME_SP:
                self._landing_pad_first_seen_time = None
                self._landing_pad_lost_time = None
                self._UKF_start = False
                self.get_logger().info("Landing Pad Lost!")
                self.controller_state = 2000

        # ---- State 2100 (Maintain Landing Pad Visual Lock - Yaw to match target)
        if (
            self.controller_state == 2100
            and self._landing_pad_found
            and t_now - self._landing_pad_first_seen_time > self.LANDING_TIME_VISUAL_TIME_SP
        ):
            self.get_logger().info(
                f"Landing Pad visual hold ok, starting {self._landing_pad_locked_time_SP} sec timer..."
            )
            self.controller_state = 3000

        # ---- State 3000 (Move over and Maintain Landing Pad Lock)
        if (
            self.controller_state == 3000
            and self._landing_pad_found
            and t_now - self._landing_pad_first_seen_time > self._landing_pad_locked_time_SP
            and err < self.LANDING_CENTERED_ERROR_THRESHOLD
        ):
            self.get_logger().info("Landing Pad Acquired")
            self.controller_state = 4000

        # ---- State 4000 (Begin Landing Descent)
        if self.controller_state == 4000:
            self.target_alt = 0.0 # self.LANDING_HEIGHT_THRESHOLD  # land
            h = -self.landing_pad_relative_position[LP_State.PZ]

            # Fallback touchdown detection: prioritize the measured velocity impact
            # over the height-based cutoff, which can fail if the height estimate drifts.
            if self._z_touchdown_trigger:
                self._z_touchdown_trigger = False
                self.cutoff = True
                self._landed_time = t_now
                self.get_logger().warn(
                    "Touchdown fallback: sharp vertical velocity reversal detected; "
                    f"vz {self._z_touchdown_trigger_prev_velocity:.2f} -> "
                    f"{self._z_touchdown_trigger_velocity:.2f} m/s, "
                    f"az={self._z_touchdown_trigger_accel:.2f} m/s^2 "
                    f"(estimated h={h:.2f} m). Engaging throttle cutoff."
                )
                self.controller_state = 6000
            elif h <= self.LANDING_CUT_HEIGHT_MAX and self._in_termination_state():
                self.cutoff = True
                self._landed_time = t_now
                self.get_logger().info(f"Throttle Cut Engaged - h = {h:.2f}, T = {self._lead_time:.2f}")
                self.controller_state = 6000
            elif h <= self.LANDING_HEIGHT_THRESHOLD:
                self.get_logger().info(f"Landing Aborted - Trying Again, h = {h:.2f}")
                self._landing_attempts += 1
                self.controller_state = 5000

        # ---- State 5000 (Landing Aborted - Regain altitude, then try again)
        if self.controller_state == 5000:
            if self._landing_attempts > 3:
                self.get_logger().info("Too many failed attempts, returning home")
                self.controller_state = 7000
            else:
                self.target_alt = self.LANDING_RECOVERY_HEIGHT
                if -self.landing_pad_relative_position_forward_predict[LP_State.PZ] > self.target_alt - 0.5:
                    self.controller_state = 3000

        # ---- State 6000 (Landing Success - Idle Until RTL)
        if self.controller_state == 6000 and (t_now - self._landed_time) > self._idle_before_RTL_SP:
            self._landing_pad_first_seen_time = None
            self._landing_pad_lost_time = None
            self.get_logger().info("Landing Complete - Idling Done", 
                                   throttle_duration_sec=10.0)
            if not self._disarm_requested:
                self._request_disarm()
            # self.controller_state = 7000 (Done!)

        # ---- State 7000 (Initiate RTL)
        if self.controller_state == 7000:
            self._initiate_rtl("Returning Home")

        # Get dt
        now = self.get_clock().now()
        if self._pid_last_control_time is None:
            control_dt = 1.0 / self.CTRL_FREQ
        else:
            control_dt = (now - self._pid_last_control_time).nanoseconds * 1e-9
        self._pid_last_control_time = now

        # ---- Run Controller ----
        control_stamp = now.to_msg()

        # Only pass through marker yaw if uncertainty on yaw is low enough
        if self._UKF_diag["sigma_yaw"] <= self.PID_MARKER_YAW_SIGMA_THRESHOLD:
            marker_yaw = self.landing_pad_yaw
        else:
            marker_yaw = None

        if self.controller_state == 1200:
            self.alt_pos_control = True
            msg = self._pid_controller.goPosition(control_stamp, **self.gps_target, yaw=self.quad_yaw)
            self.global_pos_pub.publish(msg)
        elif 2000 <= self.controller_state < 3000:
            self.alt_pos_control = True
            msg = self._pid_controller.stop(control_stamp)
            self.vel_pub.publish(msg)
        elif 3000 <= self.controller_state <= 6000:
            # Descent (4000) commands a climb rate directly; everything else holds height.
            # 6000 keeps the last rate (the cutoff flag drives the command there).
            self.alt_pos_control = self.controller_state != 4000
            if self.controller_state < 6000:
                close = (
                    self.controller_state == 4000
                    and -self.landing_pad_relative_position[LP_State.PZ] <= 2.0
                )
                self.target_alt_rate = self.DESCENT_RATE_CLOSE_SP if close else self.DESCENT_RATE_FAR_SP
            msg = self._pid_controller.update(
                dt=control_dt,
                target_altitude=self.target_alt,
                target_desc_rate=self.target_alt_rate,
                marker_yaw=marker_yaw,
                cutoff=self.cutoff,
                alt_pos_control=self.alt_pos_control,
                quad_yaw=self.quad_yaw,
                quad_vel=self.quad_vel,
                landing_pad_relative_position=self.landing_pad_relative_position_forward_predict,
                landing_pad_relative_velocity=self.landing_pad_relative_velocity_forward_predict,
            )
            self.att_pub.publish(msg)

        # Log diagnostics
        if self.DIAGNOSTICS_ENABLED:
            self.log_diagnostics()

        # Check for timer overruns
        end = self.get_clock().now()
        elapsed = (end - start).nanoseconds / 1e6  # ms
        if elapsed > 1.0/self.CTRL_FREQ * 1000:
            self.get_logger().warn(f"Control loop took {elapsed:.2f} ms!")


    # region --- SAFETY LOOP ----
    def _safety_loop(self):
        """Check safety conditions and initiate RTL if necessary"""
        if self._rtl_initiated or not self._tko_reached or self._disarmed:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check timer after takeoff
        if self._tko_complete_time is not None:
            elapsed_since_takeoff = current_time - self._tko_complete_time
            if elapsed_since_takeoff >= self.MAX_RUNTIME:
                self.get_logger().warn(
                    f"{self.MAX_RUNTIME} seconds elapsed since takeoff - Initiating RTL"
                )
                self._initiate_rtl(f"{self.MAX_RUNTIME}-second timer expired")
                return

        # Check boundary conditions
        x = self.quad_pose[QUAD_State.X]
        y = self.quad_pose[QUAD_State.Y]

        if abs(x) > self.BOUNDARY_LIMIT or abs(y) > self.BOUNDARY_LIMIT:
            self.get_logger().warn(
                f"Boundary violation: position ({x:.1f}, {y:.1f}) - Initiating RTL"
            )
            self._initiate_rtl(f"boundary violation at ({x:.1f}, {y:.1f})")
            return
        
        # Check UKF health (when not landed)
        if (self.controller_state < 6000 and self._UKF_diag["covar_max_eig"] > self.UKF_UNHEALTHY_COVAR):
            # Counter for 3 loops
            self._UKF_unhealthy_counter += 1
            if self._UKF_unhealthy_counter >= 3:
                self.get_logger().warn(
                    f"UKF: max eigenvalue {self._UKF_diag['covar_max_eig']:.3f} — filter diverging - aborting",
                    throttle_duration_sec=1.0
                )
                self._initiate_rtl("Landing Estimate too Bad")
        elif (self.controller_state < 6000 and self._UKF_diag["covar_max_eig"] <= self.UKF_UNHEALTHY_COVAR):
            self._UKF_unhealthy_counter = 0

    # region ---- TERMINATION STATE ----
    def _in_termination_state(self) -> bool:
        """ True if cutting the throttle now still lands the gear on the pad, even if the
            car does a worst-case brake/accelerate/turn during the fall (lead time).
        """
        T = self._lead_time
        yaw = self.landing_pad_yaw_forward_predict
        ex, ey = self.landing_pad_relative_position_forward_predict[:2]

        # Forward-predicted offset in the pad frame (x along pad heading, y lateral)
        e_long = abs(np.cos(yaw) * ex + np.sin(yaw) * ey)
        e_lat = abs(-np.sin(yaw) * ex + np.cos(yaw) * ey)

        # Worst-case extra pad drift during the fall
        drift_long = 0.5 * self.LANDING_CAR_A_MAX * T**2
        drift_lat = 0.5 * self.LANDING_CAR_A_LAT_MAX * T**2

        # Largest offset we can tolerate on each axis (<= 0 means impossible at this T)
        max_long = 0.5 * self.LANDING_PAD_SIZE[0] - drift_long - 0.5 * self.LANDING_GEAR_SIZE[0] - self.LANDING_CUT_MARGIN
        max_lat = 0.5 * self.LANDING_PAD_SIZE[1] - drift_lat - 0.5 * self.LANDING_GEAR_SIZE[1] - self.LANDING_CUT_MARGIN

        in_termination = e_long <= max_long and e_lat <= max_lat

        # Current (non-predicted) offset, for comparison with the forward-predicted one
        cx, cy = self.landing_pad_relative_position[:2]
        self.get_logger().info(
            f"Termination {'PASS' if in_termination else 'FAIL'} | T={T:.2f}s | "
            f"long: err={e_long:.2f} max={max_long:.2f} (drift {drift_long:.2f}) | "
            f"lat: err={e_lat:.2f} max={max_lat:.2f} (drift {drift_lat:.2f}) | "
            f"fwd=({ex:.2f}, {ey:.2f}) now=({cx:.2f}, {cy:.2f})",
            throttle_duration_sec=0.1,
        )
        return in_termination
        
    
    # region ---- DIAGNOSTICS FUNCTION CALLBACKS ----
    def _landing_pad_true_odometry_callback(self, msg: Odometry):
        """SITL ONLY: Pass true odometry of landing pad for data."""
        # Rotate velocities from base_link into the world frame.
        q = msg.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        R00 = 1.0 - 2.0 * (qy*qy + qz*qz)
        R01 = 2.0 * (qx*qy - qz*qw)
        R02 = 2.0 * (qx*qz + qy*qw)
        R10 = 2.0 * (qx*qy + qz*qw)
        R11 = 1.0 - 2.0 * (qx*qx + qz*qz)
        R12 = 2.0 * (qy*qz - qx*qw)
        R20 = 2.0 * (qx*qz - qy*qw)
        R21 = 2.0 * (qy*qz + qx*qw)
        R22 = 1.0 - 2.0 * (qx*qx + qy*qy)

        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        vx, vy, vz = v.x, v.y, v.z
        wx, wy, wz = w.x, w.y, w.z

        # Overwrite the original velocities with world-frame velocities.
        v.x = R00*vx + R01*vy + R02*vz
        v.y = R10*vx + R11*vy + R12*vz
        v.z = R20*vx + R21*vy + R22*vz

        w.x = R00*wx + R01*wy + R02*wz
        w.y = R10*wx + R11*wy + R12*wz
        w.z = R20*wx + R21*wy + R22*wz

        self.landing_pad_true_odometry = msg
        self._pad_true_recv = self.get_clock().now().nanoseconds * 1e-9


    def _true_odometry_callback(self, msg: Odometry):
        """SITL ONLY: Pass true odometry of the quad for data logging."""
        # Rotate velocities from base_link into the world frame.
        q = msg.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        R00 = 1.0 - 2.0 * (qy*qy + qz*qz)
        R01 = 2.0 * (qx*qy - qz*qw)
        R02 = 2.0 * (qx*qz + qy*qw)
        R10 = 2.0 * (qx*qy + qz*qw)
        R11 = 1.0 - 2.0 * (qx*qx + qz*qz)
        R12 = 2.0 * (qy*qz - qx*qw)
        R20 = 2.0 * (qx*qz - qy*qw)
        R21 = 2.0 * (qy*qz + qx*qw)
        R22 = 1.0 - 2.0 * (qx*qx + qy*qy)

        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        vx, vy, vz = v.x, v.y, v.z
        wx, wy, wz = w.x, w.y, w.z

        # Overwrite the original velocities with world-frame velocities.
        v.x = R00*vx + R01*vy + R02*vz
        v.y = R10*vx + R11*vy + R12*vz
        v.z = R20*vx + R21*vy + R22*vz

        w.x = R00*wx + R01*wy + R02*wz
        w.y = R10*wx + R11*wy + R12*wz
        w.z = R20*wx + R21*wy + R22*wz

        self.quad_true_odometry = msg
        self._quad_true_recv = self.get_clock().now().nanoseconds * 1e-9


    def _pipeline_timing_callback(self, msg: Vector3Stamped) -> None:
        """ Log latency across vision pipeline into .csv
        """
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._pipeline_timing_buffer[key] = msg
        if len(self._pipeline_timing_buffer) > 50:
            self._pipeline_timing_buffer.pop(next(iter(self._pipeline_timing_buffer)))


    def _yolo_pipeline_timing_callback(self, msg: Vector3Stamped) -> None:
        """ Log latency across YOLO vision pipeline into .csv
        """
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._yolo_pipeline_timing_buffer[key] = msg
        if len(self._yolo_pipeline_timing_buffer) > 50:
            self._yolo_pipeline_timing_buffer.pop(next(iter(self._yolo_pipeline_timing_buffer)))


    def start_diagnostics(self):
        """ Start diagnostic logging and creation of .csv file. ALso creates the true 
            odometry subscriptions to obtain ground truth data.
        """

        # Establish QOS for subscriptons for quadcopter and landing pad ground truth
        _odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Only start if ground truth is available
        if self.GROUND_TRUTH_AVAILABLE:
            self._true_odometry_sub = self.create_subscription(
                Odometry, "/quadcopter/true_odom", self._true_odometry_callback, _odom_qos
            )
            self._landing_pad_true_odometry_sub = self.create_subscription(
                Odometry,
                "/landing_pad/odom",
                self._landing_pad_true_odometry_callback,
                _odom_qos,
            )
        
        self._pipeline_timing_sub = self.create_subscription(
            Vector3Stamped, 
            "/landing_pad/pipeline_timing", 
            self._pipeline_timing_callback, 
            _odom_qos
        )
        self._pipeline_timing_buffer: dict[tuple[int, int], Vector3Stamped] = {}
        self._yolo_pipeline_timing_sub = self.create_subscription(
            Vector3Stamped, 
            "/landing_pad/yolo_pipeline_timing", 
            self._yolo_pipeline_timing_callback, 
            _odom_qos
        )
        self._yolo_pipeline_timing_buffer: dict[tuple[int, int], Vector3Stamped] = {}

        self.landing_pad_relative_odometry_pub = self.create_publisher(
            Odometry, "/landing_pad/rel_odom", 10
        )

        self.landing_pad_relative_raw_pub = self.create_publisher(
            Vector3Stamped, "/landing_pad/rel_raw", 10
        )

        # Create .csv file for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_filename = f"controller_{timestamp}.csv"
        self._csv_file = open(self._csv_filename, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "timestamp",
                # Drone data
                "quad_x",
                "quad_y",
                "quad_z",
                "quad_vx",
                "quad_vy",
                "quad_vz",
                "quad_yaw",
                "quad_true_x",
                "quad_true_y",
                "quad_true_z",
                "quad_true_vx",
                "quad_true_vy",
                "quad_true_vz",
                "quad_true_yaw",

                # Landing pad relative data
                "landing_pad_april_raw_stamp",
                "landing_pad_april_rel_x_raw",
                "landing_pad_april_rel_y_raw",
                "landing_pad_april_rel_z_raw",
                "landing_pad_april_rel_yaw_raw",
                "landing_pad_yolo_car_raw_stamp",
                "landing_pad_yolo_car_rel_x_raw",   # fallback (fitted) position, NaN in box mode
                "landing_pad_yolo_car_rel_y_raw",
                "landing_pad_yolo_car_rel_z_raw",
                "yolo_box_mode",                    # 1 = pixel-space box update, 0 = fallback fit
                "yolo_box_u_min",                   # box edges (px), NaN in fallback mode
                "yolo_box_u_max",
                "yolo_box_v_min",
                "yolo_box_v_max",
                "landing_pad_rel_x",
                "landing_pad_rel_y",
                "landing_pad_rel_z",
                "landing_pad_rel_vx",
                "landing_pad_rel_vy",
                "landing_pad_rel_vz",
                "landing_pad_rel_yaw",
                "landing_pad_rel_true_x",
                "landing_pad_rel_true_y",
                "landing_pad_rel_true_z",
                "landing_pad_rel_true_vx",
                "landing_pad_rel_true_vy",
                "landing_pad_rel_true_vz",
                "landing_pad_rel_true_yaw",

                # Landing pad global data
                "landing_pad_april_glob_x_raw",
                "landing_pad_april_glob_y_raw",
                "landing_pad_april_glob_z_raw",
                "landing_pad_april_glob_yaw_raw",
                "landing_pad_yolo_car_glob_x_raw",
                "landing_pad_yolo_car_glob_y_raw",
                "landing_pad_yolo_car_glob_z_raw",
                "landing_pad_glob_x",
                "landing_pad_glob_y",
                "landing_pad_glob_z",
                "landing_pad_glob_vx",
                "landing_pad_glob_vy",
                "landing_pad_glob_vz",
                "landing_pad_glob_a",
                "landing_pad_glob_yaw",
                "landing_pad_glob_yaw_rate",
                "landing_pad_glob_true_x",
                "landing_pad_glob_true_y",
                "landing_pad_glob_true_z",
                "landing_pad_glob_true_vx",
                "landing_pad_glob_true_vy",
                "landing_pad_glob_true_vz",
                "landing_pad_glob_true_yaw",

                # UKF covariance diagnostics
                "sigma_px",
                "sigma_py",
                "sigma_pz",
                "sigma_v",
                "sigma_a",
                "sigma_yaw",
                "sigma_yaw_rate",
                "covar_trace",
                "covar_det_log",
                "covar_max_eig",
                "nis",
                "rejected_measurements",
                "nees",

                # PID diagnostics
                "F_x", "F_y", "F_z", "F_thrust",
                "climb_rate_cmd", "thrust_field", "roll_cmd", "pitch_cmd", "yaw_cmd",
                "accel_perp_x", "accel_perp_y", "accel_par_x", "accel_par_y",
                "z_err", "gain_factor", "lam",
                "sat_climb_cmd", "sat_roll_clamp", "sat_pitch_clamp",
                "slew_roll", "slew_pitch", "slew_yaw",
                "marker_yaw_valid", "cutoff_active",

                # Latency through pipeline
                "cam_to_image_lag",
                "image_to_transform_lag",
                "transform_to_UKF_lag",
                "yolo_cam_to_image_lag",
                "yolo_image_to_transform_lag",
                "yolo_transform_to_UKF_lag",
                "total_lag",

                # Orchestrator state / outcome
                "controller_state", "pad_found", "pad_lost_for", "cutoff",
                "landing_attempts", "rtl", "rtl_reason",
                "target_alt", "target_alt_rate", "alt_pos_control",

                # What the controller actually consumed
                "est_time", "fwd_horizon",
                "fwd_rel_x", "fwd_rel_y", "fwd_rel_z",
                "fwd_rel_vx", "fwd_rel_vy", "fwd_rel_vz",
                "fwd_yaw", "fwd_err_xy", "fwd_alt",

                # Attitude (FCU estimate)
                "quad_roll", "quad_pitch",

                # Ground truth timing and height above pad
                "quad_true_stamp", "quad_true_recv",
                "pad_true_stamp", "pad_true_recv", "true_height",

                # Covariance off-diagonals for offline NEES
                "P_px_py", "P_px_yaw", "P_py_yaw",
            ]
        )
        self.get_logger().info(f"CSV logging initialized: {self._csv_filename}")


    def log_diagnostics(self):
        """ Log data from controller into .csv file. 
        """

        # UKF Diagnostics
        d = self._UKF_diag  # shorthand
        m = self._UKF_filter.get_measurement_diagnostics()
        apr = m["apriltag"]
        yolo = m["yolo"]
        latest = m["latest"]

        # YOLO raw is a fitted position (fallback) or box edges in pixels (box mode)
        yolo_pos = np.full(3, np.nan) if yolo["box"] else yolo["raw"][:3]
        yolo_box = yolo["raw"] if yolo["box"] else np.full(4, np.nan)

        # PID Diagnostics
        p = self._pid_controller.get_control_outputs()

        # For live plotting publishing
        self.landing_pad_relative_odometry_pub.publish(self.landing_pad_relative_odometry)

        UKF_raw_msg = Vector3Stamped()
        UKF_raw_msg.header.stamp = self.landing_pad_relative_odometry.header.stamp # Compare at same timestamp
        UKF_raw_msg.header.frame_id = "local"
        UKF_raw_msg.vector.x = apr["raw"][LP_Meas.PX]
        UKF_raw_msg.vector.y = apr["raw"][LP_Meas.PY]
        UKF_raw_msg.vector.z = apr["raw"][LP_Meas.PZ]
        self.landing_pad_relative_raw_pub.publish(UKF_raw_msg)

        current_time = self.get_clock().now().nanoseconds / 1e9

        # ---- Extra diagnostics (shared by both branches) ----
        fp = self.landing_pad_relative_position_forward_predict
        fv = self.landing_pad_relative_velocity_forward_predict
        Pc = self._UKF_filter.P

        if self.GROUND_TRUTH_AVAILABLE:
            qs = self.quad_true_odometry.header.stamp
            ps = self.landing_pad_true_odometry.header.stamp
            true_cols = [
                qs.sec + qs.nanosec * 1e-9, self._quad_true_recv,
                ps.sec + ps.nanosec * 1e-9, self._pad_true_recv,
                self.landing_pad_true_odometry.pose.pose.position.z
                - self.quad_true_odometry.pose.pose.position.z,   # true_height
            ]
        else:
            true_cols = [np.nan] * 5

        extra = [
            self.controller_state, int(self._landing_pad_found),
            (current_time - self._landing_pad_lost_time)
            if self._landing_pad_lost_time is not None else 0.0,
            int(self.cutoff), self._landing_attempts,
            int(self._rtl_initiated), self._rtl_reason,
            self.target_alt, self.target_alt_rate, int(self.alt_pos_control),

            self._UKF_est_time, self._UKF_fwd_horizon,
            fp[0], fp[1], fp[2],
            fv[0], fv[1], fv[2],
            self.landing_pad_yaw_forward_predict,
            float(np.hypot(fp[0], fp[1])),   # the value the abort check uses
            float(-fp[2]),

            self.quad_roll, self.quad_pitch,
        ] + true_cols + [
            Pc[LP_State.PX, LP_State.PY],
            Pc[LP_State.PX, LP_State.YAW],
            Pc[LP_State.PY, LP_State.YAW],
        ]

        # ---- Ground truth (NaN when unavailable, e.g. real flights) ----
        def odom_arrays(o):
            pp, tw, q = o.pose.pose.position, o.twist.twist.linear, o.pose.pose.orientation
            yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            return np.array([pp.x, pp.y, pp.z]), np.array([tw.x, tw.y, tw.z]), yaw

        if self.GROUND_TRUTH_AVAILABLE:
            q_pos, q_vel, quad_true_yaw = odom_arrays(self.quad_true_odometry)
            p_pos, p_vel, pad_true_yaw = odom_arrays(self.landing_pad_true_odometry)

            # NEES over (px, py, yaw); acceleration and yaw rate are masked out
            nees_states = [LP_State.PX, LP_State.PY, LP_State.YAW]
            e = np.array([*(p_pos - q_pos)[:2], pad_true_yaw]) - self._UKF_filter.x[nees_states]
            e[-1] = self._UKF_filter._wrap(e[-1])
            try:
                nees = float(e @ np.linalg.solve(Pc[np.ix_(nees_states, nees_states)], e))
            except np.linalg.LinAlgError:
                nees = np.nan
        else:
            q_pos = q_vel = p_pos = p_vel = np.full(3, np.nan)
            quad_true_yaw = pad_true_yaw = nees = np.nan

        # ---- Estimate (relative pose and relative velocity) ----
        o = self.landing_pad_relative_odometry
        est_pos = np.array([o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z])
        est_vel = np.array([o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z])
        x = self._UKF_filter.x

        self._csv_writer.writerow(
            [
                current_time,
                # Drone data
                *self.quad_pose, *self.quad_vel, self.quad_yaw,
                *q_pos, *q_vel, quad_true_yaw,

                # Landing pad relative data
                apr["raw_stamp"], *apr["raw"][:3], apr["raw"][LP_Meas.YAW] - quad_true_yaw,
                yolo["raw_stamp"], *yolo_pos, int(yolo["box"]), *yolo_box,
                *est_pos, *est_vel, self.landing_pad_yaw - quad_true_yaw,
                *(p_pos - q_pos), *(p_vel - q_vel), pad_true_yaw - quad_true_yaw,

                # Landing pad global data
                *(apr["raw"][:3] + q_pos), apr["raw"][LP_Meas.YAW],
                *(yolo_pos + q_pos),
                *(est_pos + q_pos), *(est_vel + q_vel),
                x[LP_State.A], self.landing_pad_yaw, x[LP_State.YAW_RATE],
                *p_pos, *p_vel, pad_true_yaw,

                # UKF covariance diagnostics
                *[d[k] for k in (
                    "sigma_px", "sigma_py", "sigma_pz", "sigma_v", "sigma_a",
                    "sigma_yaw", "sigma_yaw_rate", "covar_trace", "covar_det_log",
                    "covar_max_eig", "nis", "rejected_measurements",
                )],
                nees,

                # PID diagnostics (same order as the header)
                *[p[k] for k in CONTROL_OUTPUT_KEYS],

                # Latency through pipeline
                apr["cam_to_image_lag"],
                apr["image_to_transform_lag"],
                apr["transform_to_UKF_lag"],
                yolo["cam_to_image_lag"],
                yolo["image_to_transform_lag"],
                yolo["transform_to_UKF_lag"],
                latest["total_lag"],
            ] + extra
        )


    def destroy_node(self):
        """Clean up CSV file when node is destroyed"""
        if hasattr(self, "_csv_file"):
            self._csv_file.close()
            self.get_logger().info(f"CSV file closed: {self._csv_filename}")
        super().destroy_node()


# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()