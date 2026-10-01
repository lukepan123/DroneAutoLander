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
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import GlobalPositionTarget
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandTOL
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import MessageInterval
from datetime import datetime

from .state_definitions import QUAD_State
from .state_definitions import LP_State
from .state_definitions import LP_Meas
from .ukf import UKF
from .pid_controller import PIDController

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
        self.declare_parameter("landing_height_threshold", 0.4)
        self.LANDING_HEIGHT_THRESHOLD = self.get_parameter("landing_height_threshold").get_parameter_value().double_value
        self.declare_parameter("landing_error_threshold", 0.10)
        self.LANDING_ERROR_THRESHOLD = self.get_parameter("landing_error_threshold").get_parameter_value().double_value
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

        # ---- Global State Variables ----
        self.controller_state = 0
        self.fcu_state = State()
        self.odometry  = Odometry()
        self.quad_pose = np.zeros(3)
        self.quad_vel  = np.zeros(3)
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
        self._landing_pad_found = False
        self._landing_pad_first_seen_time = None

        # ---- State 3xxx (Maintaining Landing Pad Lock) Variables ----
        self._landing_pad_locked_time_SP = self.LANDING_TIME_VISUAL_TIME_SP + self.LANDING_PAD_LOCK_TIME_SP
        self._landing_pad_lost_time = None

        # ---- State 4xxx (Beginning Landing Descent) Variables ----
        self._landed_time = None
        self.cutoff = False

        # ---- State 5xxx (Landing Aborted) Variables ----
        self._landing_attempts = 0

        # ---- State 6xxx (Landing Confirmed) Variables ----
        self._idle_before_RTL_SP = 5.0

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

        self.declare_parameter('PID_Kp_pos_z', 0.2)
        self.PID_KP_POS_Z = self.get_parameter("PID_Kp_pos_z").get_parameter_value().double_value
        self.declare_parameter('PID_Kp_vel_z', 2.0)
        self.PID_KP_VEL_Z = self.get_parameter("PID_Kp_vel_z").get_parameter_value().double_value
        self.declare_parameter('PID_Ki_vel_z', 0.2)
        self.PID_KI_VEL_Z = self.get_parameter("PID_Ki_vel_z").get_parameter_value().double_value
        self.declare_parameter('PID_vel_z_i_clamp', 2.0)
        self.PID_VEL_Z_I_CLAMP = self.get_parameter("PID_vel_z_i_clamp").get_parameter_value().double_value
        self.declare_parameter('PID_pos_vel_err_limit', 1.5)
        self.PID_POS_VEL_ERR_LIMIT = self.get_parameter("PID_pos_vel_err_limit").get_parameter_value().double_value

        self.declare_parameter('PID_mass', 1.98)
        self.PID_MASS = self.get_parameter("PID_mass").get_parameter_value().double_value
        self.declare_parameter('PID_max_thrust', 40.0)
        self.PID_MAX_THRUST = self.get_parameter("PID_max_thrust").get_parameter_value().double_value
        self.declare_parameter('PID_gravity', 9.81)
        self.PID_GRAVITY = self.get_parameter("PID_gravity").get_parameter_value().double_value
        self.declare_parameter('PID_drag_coefficient', 0.002)
        self.PID_DRAG_COEFFICIENT = self.get_parameter("PID_drag_coefficient").get_parameter_value().double_value

        self.declare_parameter('PID_max_throttle_rate', 1.0)
        self.PID_MAX_THROTTLE_RATE = self.get_parameter("PID_max_throttle_rate").get_parameter_value().double_value
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

        self.declare_parameter('PID_drop_off_strength', 0.5)
        self.PID_DROP_OFF_STRENGTH = self.get_parameter("PID_drop_off_strength").get_parameter_value().double_value
        self.declare_parameter('PID_terminal_gain', 1.0)
        self.PID_TERMINAL_GAIN = self.get_parameter("PID_terminal_gain").get_parameter_value().double_value
        self.declare_parameter('PID_drop_off', 7.0)
        self.PID_DROP_OFF = self.get_parameter("PID_drop_off").get_parameter_value().double_value
        self.declare_parameter('PID_accel_z_limit_g', 1.0)
        self.PID_ACCEL_Z_LIMIT_G = self.get_parameter("PID_accel_z_limit_g").get_parameter_value().double_value

        # region ---- INITIALISATIONS ----
        self._UKF_start = False
        self._UKF_seed_window_min_dt = self.LANDING_TIME_VISUAL_TIME_SP - 0.1

        self._UKF_last_update = self.get_clock().now()
        self._UKF_forward_predict_x = None

        self._UKF_filter = UKF(
            self.UKF_INIT_P_DIAG_, self.UKF_INIT_Q_DIAG_,
            self.UKF_ALPHA, self.UKF_BETA, self.UKF_KAPPA,
            self.UKF_MAHALANOBIS_THRESH,
            self.INIT_LP_POS, self.INIT_LP_VEL, self.INIT_LP_YAW,
            self.UKF_SEED_WINDOW_N, self._UKF_seed_window_min_dt
        )

        self._UKF_diag = self._UKF_filter.get_covar_diagnostics()
        self._UKF_unhealthy_counter = 0

        self._pid_last_control_time = None
        self._pid_controller = PIDController(
            lam_0=self.PID_LAM_0,
            Kp_0=self.PID_KP_0,
            Kd_0=self.PID_KD_0,
            Kp_pos_z=self.PID_KP_POS_Z,
            Kp_vel_z=self.PID_KP_VEL_Z,
            Ki_vel_z=self.PID_KI_VEL_Z,
            vel_z_i_clamp=self.PID_VEL_Z_I_CLAMP,
            m=self.PID_MASS,
            max_thrust=self.PID_MAX_THRUST,
            g=self.PID_GRAVITY,
            cD=self.PID_DRAG_COEFFICIENT,
            max_throttle_rate=self.PID_MAX_THROTTLE_RATE,
            max_angle_rate=self.PID_MAX_ANGLE_RATE,
            d_blend_start=self.PID_D_BLEND_START,
            d_blend_end=self.PID_D_BLEND_END,
            d_hold_radius=self.PID_D_HOLD_RADIUS,
            drop_off_strength=self.PID_DROP_OFF_STRENGTH,
            terminal_gain=self.PID_TERMINAL_GAIN,
            drop_off=self.PID_DROP_OFF,
            accel_z_limit_g=self.PID_ACCEL_Z_LIMIT_G,
            pos_vel_err_limit=self.PID_POS_VEL_ERR_LIMIT,
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
        self.home_pose = self.quad_pose.copy()

        # For some reason velocity is negative (invert so that up = +, down= -)
        self.quad_vel = np.array(
            [
                self.odometry.twist.twist.linear.x,
                self.odometry.twist.twist.linear.y,
                -self.odometry.twist.twist.linear.z,
            ]
        )

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
        """Obtain landing pad found signal from landing_pad_detector node.
        
        :param msg: Bool msg from landing_pad_detector node
        """

        self._landing_pad_found = msg.data


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


    # region ---- UKF LOOP ----
    def _UKF_loop(self):
        """ Run UKF predict and update loops. Manages state estimation of the landing
            pad platform. Calls upon both the apriltag and YOLO measurements
        """

        # Latch start of UKF for landing pad on acquisition of landing pad from detector
        # node, if its been too long since we've seen the landing pad, reset the UKF
        if self._landing_pad_found and not self._UKF_start and self.controller_state >= 2000:
            self._UKF_filter.reset()
            self._UKF_start = True

        # Get timestamp from last cycle
        now = self.get_clock().now()
        dt = (now - self._UKF_last_update).nanoseconds * 1e-9
        self._UKF_last_update = now
        now_sec = now.nanoseconds * 1e-9

        # Predict UKF step (after first measurement)
        if self._UKF_start and not self._UKF_filter.seed_pending:
            self._UKF_filter.predict(self.quad_vel, dt, now_sec)

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
        # bounding box becomes a worse estimate of the landing pad.
        alt = -self.landing_pad_relative_position_forward_predict[LP_State.PZ]

        d_near, d_far = 0.0, self.LANDING_HEIGHT_ABOVE_GND + 2.5
        scale_max, gamma = 1.5, 2.0
        scale_t = np.clip((d_far - alt) / (d_far - d_near), 0.0, 1.0)
        yolo_lp_scale = 1.0 + (scale_max - 1.0) * scale_t ** gamma

        d_far = self.LANDING_HEIGHT_ABOVE_GND + 5.0
        scale_max = 2.0
        scale_t = np.clip((d_far - alt) / (d_far - d_near), 0.0, 1.0)
        yolo_car_scale = 1.0 + (scale_max - 1.0) * scale_t ** gamma

        R_yolo_lp = np.diag([
            0.100 * yolo_lp_scale,
            0.100 * yolo_lp_scale,
            1.000,
            1e9,
        ])

        R_yolo_car = np.diag([
            0.500 * yolo_car_scale,
            0.500 * yolo_car_scale,
            1.000,
            1e9,
        ])

        # Update each measurement stream through the same UKF path.
        timing_buffers = (
            self._pipeline_timing_buffer,
            self._yolo_pipeline_timing_buffer,
        ) if self.DIAGNOSTICS_ENABLED else (None, None)

        for measurement_type, frame, R, timing_buffer in (
            ("apriltag", "landing_pad_link", R_apriltag, timing_buffers[0]),
            ("yolo_lp", "landing_pad_link_yolo", R_yolo_lp, timing_buffers[1]),
            ("yolo_car", "car_link_yolo", R_yolo_car, timing_buffers[1]),
        ):
            try:
                tf_msg = self._tf_map_landing_pad_buffer.lookup_transform(
                    "local", frame, Time()
                )

                pipeline_timing = None
                if timing_buffer is not None:
                    key = (tf_msg.header.stamp.sec, tf_msg.header.stamp.nanosec)
                    pipeline_timing = timing_buffer.pop(key, None)

                accepted = self._UKF_filter.update(
                    tf_msg,
                    measurement_type,
                    now_sec,
                    R,
                    pipeline_timing,
                )
                if not accepted:
                    self.get_logger().warn(
                        f"UKF {measurement_type} update failed!",
                        throttle_duration_sec=2.0,
                    )
            except Exception as e:
                self.get_logger().warn(
                    f"UKF {measurement_type} measurement skipped: {e}",
                    throttle_duration_sec=2.0,
                )

        # ---- Grab per-state covariance diagnostics every tick ----
        self._UKF_diag = self._UKF_filter.get_covar_diagnostics()

        # Always publish current UKF state
        x = self._UKF_filter.x

        # Forward predict by the time it would take for the drone to drop from its
        # altitude to the landing pad. This is the value used by the controller
        # Added a fudge factor (approx the delay in the img feed (20ms))
        t = 0.02 + np.sqrt(2 * 9.81 * self.LANDING_HEIGHT_THRESHOLD) / 9.81
        self._UKF_est_time = now_sec   # instant the estimate is valid at
        self._UKF_fwd_horizon = t
        self._UKF_forward_predict_x = self._UKF_filter.forward_predict(
            self.quad_vel,
            t,
        )

        self.landing_pad_relative_position_forward_predict = np.array(
            [self._UKF_forward_predict_x[LP_State.PX],
             self._UKF_forward_predict_x[LP_State.PY],
             self._UKF_forward_predict_x[LP_State.PZ]]
        )

        self.landing_pad_yaw_forward_predict = self._UKF_forward_predict_x[LP_State.YAW]
        pad_vx = self._UKF_forward_predict_x[LP_State.V] * np.cos(self.landing_pad_yaw_forward_predict)
        pad_vy = self._UKF_forward_predict_x[LP_State.V] * np.sin(self.landing_pad_yaw_forward_predict)
        pad_vz = 0.0
        self.landing_pad_velocity_forward_predict = np.array([pad_vx, pad_vy, pad_vz])

        rel_vx = pad_vx - self.quad_vel[0]
        rel_vy = pad_vy - self.quad_vel[1]
        rel_vz = -self.quad_vel[2]
        self.landing_pad_relative_velocity_forward_predict = np.array([rel_vx, rel_vy, rel_vz])

        # Publish the actual landing pad pose estimate for all other uses
        # (diagnostics etc.)
        self.landing_pad_relative_position = np.array(
            [x[LP_State.PX], x[LP_State.PY], x[LP_State.PZ]]
        )

        yaw = x[LP_State.YAW]
        pad_vx = x[LP_State.V] * np.cos(yaw)
        pad_vy = x[LP_State.V] * np.sin(yaw)
        pad_vz = 0.0
        self.landing_pad_velocity = np.array([pad_vx, pad_vy, pad_vz])

        rel_vx = pad_vx - self.quad_vel[0]
        rel_vy = pad_vy - self.quad_vel[1]
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

        # Reset logging
        self._pid_controller.reset_outputs() 

        # Calculate errors
        err_x = abs(self.landing_pad_relative_position_forward_predict[0])
        err_y = abs(self.landing_pad_relative_position_forward_predict[1])
        err = np.hypot(err_x, err_y)

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

        # ---- State 2000 (Searching for Landing Pad)
        if self.controller_state == 2000:
            if self._landing_pad_found and self._landing_pad_first_seen_time is None:
                self._landing_pad_first_seen_time = (
                    self.get_clock().now().nanoseconds / 1e9
                )
                self.get_logger().info(
                    f"Landing Pad found, starting {self.LANDING_TIME_VISUAL_TIME_SP} sec timer..."
                )
                self.controller_state = 2100

        # ---- State 2100 (Maintain Landing Pad Visual Lock - Yaw to match target)
        if self.controller_state == 2100:
            now = self.get_clock().now().nanoseconds / 1e9

            if self._landing_pad_found:
                self._landing_pad_lost_time = None
                if (now - self._landing_pad_first_seen_time) > self.LANDING_TIME_VISUAL_TIME_SP:
                    self.get_logger().info(
                        f"Landing Pad visual hold ok, starting {self._landing_pad_locked_time_SP} sec timer..."
                    )
                    self.controller_state = 3000
            else:
                if self._landing_pad_lost_time is None:
                    self._landing_pad_lost_time = now
                else:
                    if (now - self._landing_pad_lost_time) > self.LANDING_PAD_LOST_TIME_SP:
                        self._landing_pad_first_seen_time = None
                        self._landing_pad_lost_time = None
                        self.get_logger().info("Landing Pad Lost!")
                        self.controller_state = 2000

        # ---- State 3000 (Move over and Maintain Landing Pad Lock)
        if self.controller_state == 3000:
            now = self.get_clock().now().nanoseconds / 1e9

            if self._landing_pad_found:
                self._landing_pad_lost_time = None
                if (
                    now - self._landing_pad_first_seen_time
                ) > self._landing_pad_locked_time_SP and err < self.LANDING_CENTERED_ERROR_THRESHOLD:
                    self.get_logger().info("Landing Pad Acquired")
                    self.controller_state = 4000
            else:
                if self._landing_pad_lost_time is None:
                    self._landing_pad_lost_time = now
                else:
                    if (
                        now - self._landing_pad_lost_time
                    ) > self.LANDING_PAD_LOST_TIME_SP:
                        self._landing_pad_first_seen_time = None
                        self._landing_pad_lost_time = None
                        self.get_logger().info("Landing Pad Lost!")
                        self.controller_state = 2000

        # ---- State 4000 (Begin Landing Descent)
        if self.controller_state == 4000:
            now = self.get_clock().now().nanoseconds / 1e9
            self.target_alt = self.LANDING_HEIGHT_THRESHOLD  # land

            # Check we still have the target in view, reset timer if so
            if self._landing_pad_found:
                self._landing_pad_lost_time = None
            else:
                if self._landing_pad_lost_time is None:
                    self._landing_pad_lost_time = now
                else:
                    if (
                        now - self._landing_pad_lost_time
                    ) > self.LANDING_PAD_LOST_TIME_SP:
                        self._landing_pad_first_seen_time = None
                        self._landing_pad_lost_time = None
                        self.get_logger().info("Landing Pad Lost!")
                        self.controller_state = 2000

            if (
                -self.landing_pad_relative_position_forward_predict[LP_State.PZ] <= self.LANDING_HEIGHT_THRESHOLD
                and err < self.LANDING_ERROR_THRESHOLD
            ):
                self.cutoff = True
                self._landed_time = now
                self.get_logger().info(f"Throttle Cut Engaged - Err_x = {err_x}, Err_y = {err_y}")
                self.controller_state = 6000
            elif -self.landing_pad_relative_position_forward_predict[LP_State.PZ] <= self.LANDING_HEIGHT_THRESHOLD and (
                err >= self.LANDING_ERROR_THRESHOLD
            ):
                self.get_logger().info(
                    f"Landing Aborted - Trying Again Err_x = {err_x}, Err_y = {err_y}"
                )
                self._landing_attempts += 1
                self.controller_state = 5000

        # ---- State 5000 (Landing Aborted - Regain altitude)
        if self.controller_state == 5000:
            now = self.get_clock().now().nanoseconds / 1e9

            if self._landing_attempts > 3:
                self.get_logger().info("Too many failed attempts, returning home")
                self.controller_state = 7000
            else:
                # Check we still have the target in view, reset timer if so
                if self._landing_pad_found:
                    self._landing_pad_lost_time = None
                else:
                    if self._landing_pad_lost_time is None:
                        self._landing_pad_lost_time = now
                    else:
                        if (
                            now - self._landing_pad_lost_time
                        ) > self.LANDING_PAD_LOST_TIME_SP:
                            self._landing_pad_first_seen_time = None
                            self._landing_pad_lost_time = None
                            self.get_logger().info("Landing Pad Lost!")
                            self.controller_state = 2000
                
                # Regain altitude, then try again
                self.target_alt = self.LANDING_RECOVERY_HEIGHT
                if -self.landing_pad_relative_position_forward_predict[LP_State.PZ] > self.target_alt - 0.5:
                    self.controller_state = 3000

        # ---- State 6000 (Landing Success - Idle Until RTL)
        if self.controller_state == 6000:
            now = self.get_clock().now().nanoseconds / 1e9
            if (now - self._landed_time) > self._idle_before_RTL_SP:
                self._landing_pad_first_seen_time = None
                self._landing_pad_lost_time = None
                self.get_logger().info("Landing Complete - Idling Done", 
                                       throttle_duration_sec=10.0)
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
        control_stamp = self.get_clock().now().to_msg()

        # Only pass through marker yaw if uncertainty on yaw is low enough
        if self._UKF_diag["sigma_yaw"] <= self.PID_MARKER_YAW_SIGMA_THRESHOLD:
            marker_yaw = self.landing_pad_yaw_forward_predict
        else:
            marker_yaw = None

        if self.controller_state == 1200:
            self.alt_pos_control = True
            msg = self._pid_controller.goPosition(control_stamp, **self.gps_target, yaw=self.quad_yaw)
            self.global_pos_pub.publish(msg)
        elif self.controller_state >= 2000 and self.controller_state < 3000:
            self.alt_pos_control = True
            msg = self._pid_controller.stop(control_stamp)
            self.vel_pub.publish(msg)
        elif self.controller_state >= 3000 and self.controller_state < 4000:
            self.alt_pos_control = True
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
        elif self.controller_state == 4000:
            self.alt_pos_control = False
            self.target_alt_rate = self.DESCENT_RATE_FAR_SP
            if -self.landing_pad_relative_position_forward_predict[LP_State.PZ] <= 1.0:
                self.target_alt_rate = self.DESCENT_RATE_CLOSE_SP
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
        elif self.controller_state == 5000:
            self.alt_pos_control = False
            self.target_alt_rate = -self.DESCENT_RATE_CLOSE_SP
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
        elif self.controller_state > 5000 and self.controller_state <= 6000:
            self.alt_pos_control = True
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


    def _safety_loop(self):
        """Check safety conditions and initiate RTL if necessary"""
        if self._rtl_initiated or not self._tko_reached:
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

    
    # region ---- DIAGNOSTICS FUNCTION CALLBACKS ----
    def _landing_pad_true_odometry_callback(self, msg: Odometry):
        """ SITL ONLY: Pass true odometry of landing pad for data.
        """
        self.landing_pad_true_odometry = msg
        self._pad_true_recv = self.get_clock().now().nanoseconds * 1e-9


    def _true_odometry_callback(self, msg: Odometry):
        """ SITL ONLY: Pass true (ground truth) odometry of the quad itself for data 
            logging.
        """
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
                "landing_pad_yolo_LP_raw_stamp",
                "landing_pad_yolo_LP_rel_x_raw",
                "landing_pad_yolo_LP_rel_y_raw",
                "landing_pad_yolo_LP_rel_z_raw",
                "landing_pad_yolo_car_raw_stamp",
                "landing_pad_yolo_car_rel_x_raw",
                "landing_pad_yolo_car_rel_y_raw",
                "landing_pad_yolo_car_rel_z_raw",
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
                "landing_pad_yolo_LP_glob_x_raw",
                "landing_pad_yolo_LP_glob_y_raw",
                "landing_pad_yolo_LP_glob_z_raw",
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
                "throttle_cmd", "roll_cmd", "pitch_cmd", "yaw_cmd",
                "accel_perp_x", "accel_perp_y", "accel_par_x", "accel_par_y", "accel_z_cmd",
                "z_err", "vel_z_err", "vel_z_integral", "gain_factor", "lam",
                "sat_accel_z", "sat_vel_z_int", "sat_F_z", "sat_F_horiz",
                "sat_roll_clamp", "sat_pitch_clamp",
                "slew_throttle", "slew_roll", "slew_pitch", "slew_yaw",
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
        yolo_lp = m["yolo_lp"]
        yolo_car = m["yolo_car"]
        yolo = m["latest_yolo"]
        latest = m["latest"]

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

        # Only if ground-truth is available do we calculate the NEES, otherwise just 
        # log 0
        if self.GROUND_TRUTH_AVAILABLE:
            # Convert ground truth quaternions to RPY
            _, _, lp_pad_true_yaw = tf_transformations.euler_from_quaternion(
                [
                    self.landing_pad_true_odometry.pose.pose.orientation.x,
                    self.landing_pad_true_odometry.pose.pose.orientation.y,
                    self.landing_pad_true_odometry.pose.pose.orientation.z,
                    self.landing_pad_true_odometry.pose.pose.orientation.w,
                ]
            )

            _, _, true_yaw = tf_transformations.euler_from_quaternion(
                [
                    self.quad_true_odometry.pose.pose.orientation.x,
                    self.quad_true_odometry.pose.pose.orientation.y,
                    self.quad_true_odometry.pose.pose.orientation.z,
                    self.quad_true_odometry.pose.pose.orientation.w,
                ]
            )

            # Get the NEES as well
            x_true = np.array([
                self.landing_pad_true_odometry.pose.pose.position.x - self.quad_true_odometry.pose.pose.position.x,
                self.landing_pad_true_odometry.pose.pose.position.y - self.quad_true_odometry.pose.pose.position.y,
                lp_pad_true_yaw
            ])

            # Extract matching UKF states (mask acceleration and yaw rate)
            nees_states = [
                LP_State.PX,
                LP_State.PY,
                LP_State.YAW,
            ]
            x_est = self._UKF_filter.x[nees_states]

            # State error
            e = x_true - x_est
            e[-1] = self._UKF_filter._wrap(e[-1])

            # Extract matching covariance submatrix
            P_nees = self._UKF_filter.P[np.ix_(nees_states, nees_states)]

            # Compute NEES
            try:
                nees = float(e @ np.linalg.solve(P_nees, e))
            except np.linalg.LinAlgError:
                nees = np.nan

            self._csv_writer.writerow(
                [
                    current_time,
                    # Drone data
                    self.quad_pose[QUAD_State.X],
                    self.quad_pose[QUAD_State.Y],
                    self.quad_pose[QUAD_State.Z],
                    self.quad_vel[QUAD_State.X],
                    self.quad_vel[QUAD_State.Y],
                    self.quad_vel[QUAD_State.Z],
                    self.quad_yaw,
                    self.quad_true_odometry.pose.pose.position.x,
                    self.quad_true_odometry.pose.pose.position.y,
                    self.quad_true_odometry.pose.pose.position.z,
                    # Fix velocity because it is given in body frame not world frame...
                    self.quad_true_odometry.twist.twist.linear.x * np.cos(true_yaw),
                    self.quad_true_odometry.twist.twist.linear.x * np.sin(true_yaw),
                    self.quad_true_odometry.twist.twist.linear.z,
                    true_yaw,

                    # Landing pad relative data
                    apr["raw_stamp"],
                    apr["raw"][LP_Meas.PX],
                    apr["raw"][LP_Meas.PY],
                    apr["raw"][LP_Meas.PZ],
                    apr["raw"][LP_Meas.YAW] - true_yaw,
                    yolo_lp["raw_stamp"],
                    yolo_lp["raw"][LP_Meas.PX],
                    yolo_lp["raw"][LP_Meas.PY],
                    yolo_lp["raw"][LP_Meas.PZ],
                    yolo_car["raw_stamp"],
                    yolo_car["raw"][LP_Meas.PX],
                    yolo_car["raw"][LP_Meas.PY],
                    yolo_car["raw"][LP_Meas.PZ],
                    self.landing_pad_relative_odometry.pose.pose.position.x,
                    self.landing_pad_relative_odometry.pose.pose.position.y,
                    self.landing_pad_relative_odometry.pose.pose.position.z,
                    self.landing_pad_relative_odometry.twist.twist.linear.x,
                    self.landing_pad_relative_odometry.twist.twist.linear.y,
                    self.landing_pad_relative_odometry.twist.twist.linear.z,
                    self.landing_pad_yaw - true_yaw,
                    self.landing_pad_true_odometry.pose.pose.position.x - self.quad_true_odometry.pose.pose.position.x,
                    self.landing_pad_true_odometry.pose.pose.position.y - self.quad_true_odometry.pose.pose.position.y,
                    self.landing_pad_true_odometry.pose.pose.position.z - self.quad_pose[QUAD_State.Z],
                    # Fix velocity because it is given in body frame not world frame...
                    self.landing_pad_true_odometry.twist.twist.linear.x * np.cos(lp_pad_true_yaw) - self.quad_true_odometry.twist.twist.linear.x * np.cos(true_yaw),
                    self.landing_pad_true_odometry.twist.twist.linear.x * np.sin(lp_pad_true_yaw) - self.quad_true_odometry.twist.twist.linear.x * np.sin(true_yaw),
                    self.landing_pad_true_odometry.twist.twist.linear.z - self.quad_true_odometry.twist.twist.linear.z,
                    lp_pad_true_yaw - true_yaw,

                    # Landing pad global data
                    apr["raw"][LP_Meas.PX] + self.quad_true_odometry.pose.pose.position.x,
                    apr["raw"][LP_Meas.PY] + self.quad_true_odometry.pose.pose.position.y,
                    apr["raw"][LP_Meas.PZ] + self.quad_pose[QUAD_State.Z],
                    apr["raw"][LP_Meas.YAW],
                    yolo_lp["raw"][LP_Meas.PX] + self.quad_true_odometry.pose.pose.position.x,
                    yolo_lp["raw"][LP_Meas.PY] + self.quad_true_odometry.pose.pose.position.y,
                    yolo_lp["raw"][LP_Meas.PZ] + self.quad_pose[QUAD_State.Z],
                    yolo_car["raw"][LP_Meas.PX] + self.quad_true_odometry.pose.pose.position.x,
                    yolo_car["raw"][LP_Meas.PY] + self.quad_true_odometry.pose.pose.position.y,
                    yolo_car["raw"][LP_Meas.PZ] + self.quad_pose[QUAD_State.Z],
                    self.landing_pad_relative_odometry.pose.pose.position.x + self.quad_true_odometry.pose.pose.position.x,
                    self.landing_pad_relative_odometry.pose.pose.position.y + self.quad_true_odometry.pose.pose.position.y,
                    self.landing_pad_relative_odometry.pose.pose.position.z + self.quad_pose[QUAD_State.Z],
                    self.landing_pad_relative_odometry.twist.twist.linear.x + self.quad_true_odometry.twist.twist.linear.x * np.cos(true_yaw),
                    self.landing_pad_relative_odometry.twist.twist.linear.y + self.quad_true_odometry.twist.twist.linear.x * np.sin(true_yaw),
                    self.landing_pad_relative_odometry.twist.twist.linear.z + self.quad_true_odometry.twist.twist.linear.z,
                    self._UKF_filter.x[LP_State.A],
                    self.landing_pad_yaw,
                    self._UKF_filter.x[LP_State.YAW_RATE],
                    self.landing_pad_true_odometry.pose.pose.position.x,
                    self.landing_pad_true_odometry.pose.pose.position.y,
                    self.landing_pad_true_odometry.pose.pose.position.z,
                    self.landing_pad_true_odometry.twist.twist.linear.x * np.cos(lp_pad_true_yaw),
                    self.landing_pad_true_odometry.twist.twist.linear.x * np.sin(lp_pad_true_yaw),
                    self.landing_pad_true_odometry.twist.twist.linear.z,
                    lp_pad_true_yaw,   

                    # ---- UKF covariance diagnostics ----
                    d["sigma_px"],
                    d["sigma_py"],
                    d["sigma_pz"],
                    d["sigma_v"],
                    d["sigma_a"],
                    d["sigma_yaw"],
                    d["sigma_yaw_rate"],
                    d["covar_trace"],
                    d["covar_det_log"],
                    d["covar_max_eig"],
                    d["nis"],
                    d["rejected_measurements"],
                    nees,

                    # PID diagnostics
                    p["F_x"], p["F_y"], p["F_z"], p["F_thrust"],
                    p["throttle_cmd"], p["roll_cmd"], p["pitch_cmd"], p["yaw_cmd"],
                    p["accel_perp_x"], p["accel_perp_y"], p["accel_par_x"], p["accel_par_y"], p["accel_z_cmd"],
                    p["z_err"], p["vel_z_err"], p["vel_z_integral"], p["gain_factor"], p["lam"],
                    p["sat_accel_z"], p["sat_vel_z_int"], p["sat_F_z"], p["sat_F_horiz"],
                    p["sat_roll_clamp"], p["sat_pitch_clamp"],
                    p["slew_throttle"], p["slew_roll"], p["slew_pitch"], p["slew_yaw"],
                    p["marker_yaw_valid"], p["cutoff_active"],

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
        else:
            self._csv_writer.writerow(
                [
                    current_time,
                    # Drone data
                    self.quad_pose[QUAD_State.X],
                    self.quad_pose[QUAD_State.Y],
                    self.quad_pose[QUAD_State.Z],
                    self.quad_vel[QUAD_State.X],
                    self.quad_vel[QUAD_State.Y],
                    self.quad_vel[QUAD_State.Z],
                    self.quad_yaw,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,

                    # Landing pad relative data
                    apr["raw_stamp"],
                    apr["raw"][LP_Meas.PX],
                    apr["raw"][LP_Meas.PY],
                    apr["raw"][LP_Meas.PZ],
                    0,
                    yolo_lp["raw_stamp"],
                    yolo_lp["raw"][LP_Meas.PX],
                    yolo_lp["raw"][LP_Meas.PY],
                    yolo_lp["raw"][LP_Meas.PZ],
                    yolo_car["raw_stamp"],
                    yolo_car["raw"][LP_Meas.PX],
                    yolo_car["raw"][LP_Meas.PY],
                    yolo_car["raw"][LP_Meas.PZ],
                    self.landing_pad_relative_odometry.pose.pose.position.x,
                    self.landing_pad_relative_odometry.pose.pose.position.y,
                    self.landing_pad_relative_odometry.pose.pose.position.z,
                    self.landing_pad_relative_odometry.twist.twist.linear.x,
                    self.landing_pad_relative_odometry.twist.twist.linear.y,
                    self.landing_pad_relative_odometry.twist.twist.linear.z,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,

                    # Landing pad global data
                    0,
                    0,
                    0,
                    apr["raw"][LP_Meas.YAW],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    self._UKF_filter.x[LP_State.A],
                    self.landing_pad_yaw,
                    self._UKF_filter.x[LP_State.YAW_RATE],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,   

                    # ---- UKF covariance diagnostics ----
                    d["sigma_px"],
                    d["sigma_py"],
                    d["sigma_pz"],
                    d["sigma_v"],
                    d["sigma_a"],
                    d["sigma_yaw"],
                    d["sigma_yaw_rate"],
                    d["covar_trace"],
                    d["covar_det_log"],
                    d["covar_max_eig"],
                    d["nis"],
                    d["rejected_measurements"],
                    0,

                    # PID diagnostics
                    p["F_x"], p["F_y"], p["F_z"], p["F_thrust"],
                    p["throttle_cmd"], p["roll_cmd"], p["pitch_cmd"], p["yaw_cmd"],
                    p["accel_perp_x"], p["accel_perp_y"], p["accel_par_x"], p["accel_par_y"], p["accel_z_cmd"],
                    p["z_err"], p["vel_z_err"], p["vel_z_integral"], p["gain_factor"], p["lam"],
                    p["sat_accel_z"], p["sat_vel_z_int"], p["sat_F_z"], p["sat_F_horiz"],
                    p["sat_roll_clamp"], p["sat_pitch_clamp"],
                    p["slew_throttle"], p["slew_roll"], p["slew_pitch"], p["slew_yaw"],
                    p["marker_yaw_valid"], p["cutoff_active"],

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