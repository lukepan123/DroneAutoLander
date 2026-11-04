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

# ---- PIPELINE LATENCY STAGES ----
# Each stream reports end-of-stage timestamps after the image stamp (detector clock):
#   apriltag: image received, transform ready        (side topic /landing_pad/pipeline_timing)
#   yolo:     image received, processing started,
#             inference done, measurement published  (embedded in yolo_car_meas [24:28])
# The controller adds two of its own:
#   wait:  last detector timestamp -> UKF tick that consumes the measurement
#   apply: UKF tick start -> this measurement's update done (fit, OOSM replay, ...)
# and total = image stamp -> update done. All in ms, logged once per measurement.
LATENCY_STAGES = {
    "apriltag": ("transport", "processing"),
    "yolo":     ("transport", "queue", "inference", "post"),
}
LATENCY_TAIL = ("wait", "apply", "total")
LATENCY_COLUMNS = [
    f"{stream}_{stage}_lag"
    for stream, stages in LATENCY_STAGES.items()
    for stage in stages + LATENCY_TAIL
]

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
        # Shadow fit: run the yaw-agnostic box fit on every box-mode frame and log it as a
        # raw baseline (never fed to the filter). Diagnostics only: costs CPU, off in flight.
        self.declare_parameter("yolo_shadow_fit_enabled", False)
        self.YOLO_SHADOW_FIT_ENABLED = self.get_parameter("yolo_shadow_fit_enabled").get_parameter_value().bool_value
        # Real flight: never switch the FCU into GUIDED ourselves. The pilot selecting
        # GUIDED on the RC is the "go" signal; arming and takeoff then follow as in SITL.
        self.declare_parameter("wait_for_pilot_guided", False)
        self.WAIT_FOR_PILOT_GUIDED = self.get_parameter("wait_for_pilot_guided").get_parameter_value().bool_value
        # MAVLink stream rate requested for GLOBAL_POSITION_INT / ATTITUDE (Hz). 0 = leave
        # the FCU's SRx_* stream rates alone.
        self.declare_parameter("mavros_message_rate_hz", 0.0)
        self.MAVROS_MESSAGE_RATE_HZ = self.get_parameter("mavros_message_rate_hz").get_parameter_value().double_value

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
        # Only armed below this estimated height above the pad (inf = ungated, the SITL
        # behaviour). Real EKF velocity is noisy enough to fake a reversal mid-air, and a
        # false trigger here ends in a force disarm.
        self.declare_parameter("landing_z_touchdown_accel_threshold", 4.0)  # m/s^2
        self.LANDING_Z_TOUCHDOWN_ACCEL_THRESHOLD = self.get_parameter("landing_z_touchdown_accel_threshold").get_parameter_value().double_value
        self.declare_parameter("landing_z_touchdown_max_height", float("inf"))  # m
        self.LANDING_Z_TOUCHDOWN_MAX_HEIGHT = self.get_parameter("landing_z_touchdown_max_height").get_parameter_value().double_value
        self.declare_parameter("landing_centered_error_threshold", 2.0)
        self.LANDING_CENTERED_ERROR_THRESHOLD = self.get_parameter("landing_centered_error_threshold").get_parameter_value().double_value
        # State 2100 holds until the car's relative bearing is within this of the drone heading
        self.declare_parameter("landing_acquire_bearing_max_deg", 120.0)
        self.LANDING_ACQUIRE_BEARING_MAX = np.radians(self.get_parameter("landing_acquire_bearing_max_deg").get_parameter_value().double_value)
        # Early acquisition (3000 -> 4000 before the lock timer): the UKF is confident and
        # the drone is matching the pad's velocity, continuously for the dwell time
        self.declare_parameter("landing_confident_pos_sigma", 0.50)      # m, 1-sigma horizontal position
        self.LANDING_CONFIDENT_POS_SIGMA = self.get_parameter("landing_confident_pos_sigma").get_parameter_value().double_value
        self.declare_parameter("landing_confident_vel_sigma", 2.00)      # m/s, 1-sigma pad velocity vector
        self.LANDING_CONFIDENT_VEL_SIGMA = self.get_parameter("landing_confident_vel_sigma").get_parameter_value().double_value
        self.declare_parameter("landing_confident_rel_vel", 0.60)        # m/s, |horizontal relative velocity|
        self.LANDING_CONFIDENT_REL_VEL = self.get_parameter("landing_confident_rel_vel").get_parameter_value().double_value
        self.declare_parameter("landing_confident_dwell (s)", 1.0)
        self.LANDING_CONFIDENT_DWELL = self.get_parameter("landing_confident_dwell (s)").get_parameter_value().double_value

        # Landing state timers/parameters
        self.declare_parameter("landing_chase_altitude", 6.0)
        self.LANDING_CHASE_HEIGHT_SP = self.get_parameter("landing_chase_altitude").get_parameter_value().double_value
        self.target_alt = self.LANDING_CHASE_HEIGHT_SP
        self.declare_parameter("landing_descent_rate_far", -1.00)
        self.DESCENT_RATE_FAR_SP = self.get_parameter("landing_descent_rate_far").get_parameter_value().double_value
        self.target_alt_rate = self.DESCENT_RATE_FAR_SP
        self.declare_parameter("landing_descent_rate_close", -0.50)
        self.DESCENT_RATE_CLOSE_SP = self.get_parameter("landing_descent_rate_close").get_parameter_value().double_value
        self.declare_parameter("landing_time_visual_min_time (s)", 1.1)
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
        self.declare_parameter("landing_cut_margin", 0.10)
        self.LANDING_CUT_MARGIN = self.get_parameter("landing_cut_margin").get_parameter_value().double_value
        self.declare_parameter("landing_pad_size", [1.80, 1.48])
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
        self.quad_roll  = 0.0
        self.quad_pitch = 0.0
        self.quad_yaw   = 0.0
        self.landing_pad_relative_odometry = Odometry()
        self.landing_pad_acceleration = np.zeros(2)   # pad accel (level frame, xy), from the UKF
        self._pad_acc_ff = np.full(2, np.nan)         # feedforward actually applied this tick (logged)
        self.alt_pos_control = True

        self._last_quad_z_velocity = None
        self._last_quad_z_velocity_time = None
        self._z_touchdown_trigger = False
        self._z_touchdown_trigger_accel = 0.0
        self._z_touchdown_trigger_prev_velocity = 0.0
        self._z_touchdown_trigger_velocity = 0.0
        self._yolo_shadow_fit = None     # box-mode runs: fit on the same frame, logged only
        self._yolo_shadow_stamp = None
        self._yolo_fit_yaw = None        # (stamp, yaw, sigma) of the last box yaw fed to the UKF
        self._ukf_diverged = False       # set by the UKF loop on a re-seed, consumed by the control loop

        # Pipeline latency: one sample per measurement, written by the UKF loop and popped
        # by the next CSV row (so repeated rows log NaN instead of duplicating it)
        self._latency_pending = {}
        self._latency_last_stamp = dict.fromkeys(LATENCY_STAGES)

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
        self._confident_since = None     # start of the current run of early-acquisition confidence
        self._lock_timer_start = None    # entry into 3000; restarted on every retry after an abort
        self._gate_pos_sigma = np.nan    # early-acquisition gate inputs, logged for tuning
        self._gate_vel_sigma = np.nan
        self._gate_rel_speed = np.nan
        self._gate_confident = 0

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

        # ---- State 8xxx (Pilot override) Variables ----
        # Latched once the FCU leaves GUIDED without us asking (pilot mode switch or an
        # FCU failsafe). The controller then goes passive: no setpoints, no RTL, no disarm.
        self._pilot_override = False

        # region ---- UKF PARAMETERS ----
        self.declare_parameter('UKF_seeding_num_samples', 10)
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
        # Divergence guard: this many live measurements gated out in a row -> re-seed
        self.declare_parameter('UKF_max_consecutive_rejections', 5)   # ~0.5 s of YOLO frames
        self.UKF_MAX_CONSEC_REJECT = self.get_parameter("UKF_max_consecutive_rejections").get_parameter_value().integer_value
        # OOSM history: must exceed the worst image -> UKF latency (slower on the Pi)
        self.declare_parameter('UKF_buffer_window_s', 0.5)
        self.UKF_BUFFER_WINDOW = self.get_parameter("UKF_buffer_window_s").get_parameter_value().double_value

        # AprilTag fusion toggle (the AprilTag node may not be running at all)
        self.declare_parameter('apriltag_enabled', False)
        self.APRILTAG_ENABLED = self.get_parameter("apriltag_enabled").get_parameter_value().bool_value

        # YOLO pixel-space box update: used once yaw is known well enough, otherwise the
        # box fit (position, plus resolved box yaw) is fed instead, as during seeding
        self.declare_parameter('UKF_yolo_box_yaw_sigma_max', 0.25)   # rad
        self.UKF_YOLO_BOX_YAW_SIGMA_MAX = self.get_parameter("UKF_yolo_box_yaw_sigma_max").get_parameter_value().double_value
        self.declare_parameter('UKF_yolo_box_pixel_sigma', 10.0)      # px, per box edge
        self.UKF_YOLO_BOX_PIXEL_SIGMA = self.get_parameter("UKF_yolo_box_pixel_sigma").get_parameter_value().double_value

        # Box-derived yaw for the fallback fit: the box gives |car yaw - camera heading|,
        # resolved against the filter's heading (front/back, mirror) and fed as a yaw
        # measurement. Speeds up yaw convergence at low car speed, where motion says little.
        # Only used inside a speed window: below it the motion heading can't be trusted
        # for front/back, above it motion already beats the box. Also caps the seeded yaw
        # variance (see UKF yaw_seed_var_max). False = original behaviour, bit for bit.
        self.declare_parameter('UKF_yolo_fit_yaw_enabled', True)
        self.UKF_YOLO_FIT_YAW_ENABLED = self.get_parameter("UKF_yolo_fit_yaw_enabled").get_parameter_value().bool_value
        self.declare_parameter('UKF_yolo_fit_yaw_speed_window', [0.3, 10.0])  # m/s, estimated car speed
        self.UKF_YOLO_FIT_YAW_SPEED_WINDOW = self.get_parameter("UKF_yolo_fit_yaw_speed_window").get_parameter_value().double_array_value
        self.declare_parameter('UKF_yolo_fit_yaw_pixel_sigma', 3.0)   # px, YOLO edge noise behind the fitted yaw sigma
        self.UKF_YOLO_FIT_YAW_PIXEL_SIGMA = self.get_parameter("UKF_yolo_fit_yaw_pixel_sigma").get_parameter_value().double_value
        self.declare_parameter('UKF_yolo_fit_yaw_sigma_floor', 0.05)  # rad, absorbs car-model mismatch
        self.UKF_YOLO_FIT_YAW_SIGMA_FLOOR = self.get_parameter("UKF_yolo_fit_yaw_sigma_floor").get_parameter_value().double_value

        # YOLO car shape: union of boxes in the car model's frame (x forward, y left, z up,
        # same origin as the car's odometry), 6 numbers per box:
        # x_min, x_max, y_min, y_max, z_min, z_max (m). Override per car in the launch file.
        self.declare_parameter("yolo_car_boxes", [
            -2.31, 2.24, -0.89, 0.89, 0.00, 0.85,      # lower body, bumpers, wheels
            -2.25, 1.30, -0.89, 0.89, 0.85, 1.15,      # upper body, trunk deck
            -1.40, 0.45, -0.64, 0.64, 1.15, 1.38,      # greenhouse
            -1.15, 0.20, -0.64, 0.64, 1.38, 1.49,      # roof
            -1.40, 0.40, -0.74, 0.74, 1.515, 1.525,    # landing platform (1.80 x 1.48)
        ])
        self.YOLO_CAR_BOXES = self.get_parameter("yolo_car_boxes").get_parameter_value().double_array_value
        self.declare_parameter("yolo_car_rear_axle", [-1.45, 0.0])  # rear axle in the car frame (m)
        self.YOLO_CAR_REAR_AXLE = self.get_parameter("yolo_car_rear_axle").get_parameter_value().double_array_value
        # Landing pad top-centre in the same frame: the point the UKF tracks
        self.declare_parameter("yolo_car_pad_position", [-0.5, 0.0, 1.525])
        self.YOLO_CAR_PAD_POSITION = self.get_parameter("yolo_car_pad_position").get_parameter_value().double_array_value
        self.declare_parameter("yolo_car_fit_max_rel_rms", 0.08)  # max 3D-box fit residual, as a fraction of box size
        self.YOLO_CAR_FIT_MAX_REL_RMS = self.get_parameter("yolo_car_fit_max_rel_rms").get_parameter_value().double_value
        self.declare_parameter("yolo_roof_height_gain", 0.2)  # per car detection; how fast the pad height tracks (0 = fixed)
        self.YOLO_ROOF_HEIGHT_GAIN = self.get_parameter("yolo_roof_height_gain").get_parameter_value().double_value

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
        self.declare_parameter('PID_drag_coefficient', 0.03)
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
        # Pad-acceleration feedforward (centripetal + along-track), scaled by this gain. Lets
        # the PD hold the forecast cut point through turns instead of lagging outside them.
        # 0.0 = off (pure PD); reduce below 1 if the estimated a / yaw rate are noisy.
        self.declare_parameter('PID_accel_ff_gain', 1.0)
        self.PID_ACCEL_FF_GAIN = self.get_parameter("PID_accel_ff_gain").get_parameter_value().double_value

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
            self.YOLO_CAR_BOXES, self.YOLO_CAR_PAD_POSITION,
            self.YOLO_CAR_FIT_MAX_REL_RMS, self.YOLO_ROOF_HEIGHT_GAIN,
            rear_axle=self.YOLO_CAR_REAR_AXLE,
            yaw_seed_var_max=1.0 if self.UKF_YOLO_FIT_YAW_ENABLED else None,
            buffer_window=self.UKF_BUFFER_WINDOW,
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

        # Raw YOLO car box + image-time camera geometry + pipeline timestamps (layout
        # documented in yolo.py)
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
        self._csv_last_flush = 0.0
        self.landing_pad_true_odometry = Odometry()
        if self.DIAGNOSTICS_ENABLED:
            self.start_diagnostics()

        self.get_logger().info("Auto Lander (callbacks) started")


    # region ---- CALLBACK IMPLEMENTATIONS ----
    def _set_message_intervals(self) -> None:
        """ Request MAVLink stream rates (mavros_message_rate_hz > 0 only). The odometry
            rate sets how well camera poses are interpolated to image stamps and how
            fresh the velocity is that the touchdown fallback differentiates.
        """
        rate = self.MAVROS_MESSAGE_RATE_HZ
        if rate <= 0.0:
            return
        # Drives /mavros/global_position/local (position + velocity)
        self._set_single_message_interval(33, rate, "GLOBAL_POSITION_INT")
        # Drives the orientation inside that odometry (ArduPilot sends ATTITUDE, not
        # ATTITUDE_QUATERNION, by default)
        self._set_single_message_interval(30, rate, "ATTITUDE")


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
                # impact/rebound. Only a loose height gate (landing_z_touchdown_max_height,
                # inf in SITL), since the height estimate itself may have drifted.
                velocity_rising = (self._last_quad_z_velocity < z_velocity) #type: ignore

                if (
                    self.controller_state == 4000
                    and velocity_rising
                    and self.quad_z_accel >= self.LANDING_Z_TOUCHDOWN_ACCEL_THRESHOLD
                    and -self.landing_pad_relative_position[LP_State.PZ]
                    <= self.LANDING_Z_TOUCHDOWN_MAX_HEIGHT
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
        elif self.quad_pose[QUAD_State.Z] > 2.0:
            # Pilot took off manually before selecting GUIDED: ArduCopter refuses a
            # takeoff when already flying, so skip it rather than retrying forever.
            self.get_logger().warn(
                f"Takeoff rejected but already airborne at {self.quad_pose[QUAD_State.Z]:.1f} m "
                "- skipping takeoff."
            )
            if not self._tko_reached:
                self._tko_reached = True
                self._tko_complete_time = self.get_clock().now().nanoseconds / 1e9
        else:
            self.get_logger().error("Takeoff rejected by FCU.")
            self._tko_requested = False


    def _initiate_rtl(self, reason) -> None:
        """Initiate return to land mode and disable tracking.

        :param reason: Reason for RTL
        """

        if self._rtl_initiated:
            return

        # TODO: Never override a pilot (or FCU failsafe) that has taken the vehicle out of GUIDED
        if self._pilot_override or self.fcu_state.mode != self._mode:
            self.get_logger().warn(
                f"RTL ({reason}) skipped: FCU in {self.fcu_state.mode or 'unknown'}, not {self._mode}",
                throttle_duration_sec=5.0,
            )
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
        scale_max, gamma = 20.0, 2.0
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

        # Gather measurements as (type, stamp, z, R, cam, checkpoints). checkpoints are
        # the detector's end-of-stage timestamps (see LATENCY_STAGES), or None.
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
                timing = (self._pipeline_timing_buffer.pop(key, None)
                          if self.DIAGNOSTICS_ENABLED else None)
                measurements.append((
                    "apriltag", key[0] + key[1] * 1e-9, np.array([t.x, t.y, t.z, yaw]),
                    R_apriltag, None,
                    None if timing is None else (timing.vector.x, timing.vector.y),
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
            # Pipeline timestamps ride in the message itself (older yolo.py: absent)
            checkpoints = d[24:28] if d.size >= 28 else None

            sigma_yaw = np.sqrt(max(self._UKF_filter.P[LP_State.YAW, LP_State.YAW], 0.0))
            if self._UKF_filter.seed_pending or sigma_yaw >= self.UKF_YOLO_BOX_YAW_SIGMA_MAX:
                # Yaw too poor for the pixel model: box fit -> position, plus the box yaw
                # when the filter's heading can tell front/back (< 45 deg, car clearly
                # moving forward) and the car is slow enough for the box to beat motion
                v_lo, v_hi = self.UKF_YOLO_FIT_YAW_SPEED_WINDOW
                use_yaw = (self.UKF_YOLO_FIT_YAW_ENABLED and not self._UKF_filter.seed_pending
                           and sigma_yaw < np.pi / 4
                           and v_lo < self._UKF_filter.x[LP_State.V] < v_hi)
                yaw_ref = ((self._UKF_filter.x[LP_State.YAW], sigma_yaw, self.UKF_YOLO_FIT_YAW_PIXEL_SIGMA)
                           if use_yaw else None)
                fit = self._UKF_filter.fit_car_box(d[2:6], cam, yaw_ref=yaw_ref)
                if fit is not None:
                    pos, fit_yaw, fit_yaw_sigma = fit
                    z, R = np.append(pos, 0.0), R_yolo_car
                    # Skip box yaw no tighter than the filter's own: its frame-to-frame
                    # errors are correlated, so weak repeats add bias, not information
                    if fit_yaw is not None and fit_yaw_sigma < sigma_yaw:
                        z[LP_Meas.YAW] = fit_yaw
                        R = R_yolo_car.copy()
                        R[LP_Meas.YAW, LP_Meas.YAW] = max(fit_yaw_sigma, self.UKF_YOLO_FIT_YAW_SIGMA_FLOOR) ** 2 #type: ignore
                        self._yolo_fit_yaw = (stamp, fit_yaw, np.sqrt(R[LP_Meas.YAW, LP_Meas.YAW]))
                    measurements.append(("yolo", stamp, z, R, None, checkpoints))
            else:
                measurements.append(("yolo", stamp, d[2:6], R_yolo_box, cam, checkpoints))
                # Shadow fit: logged as the raw baseline, never fed to the filter
                if self.YOLO_SHADOW_FIT_ENABLED:
                    fit = self._UKF_filter.fit_car_box(d[2:6], cam, learn=False)
                    self._yolo_shadow_fit = None if fit is None else fit[0]
                    self._yolo_shadow_stamp = stamp

        # Update each measurement stream through the same UKF path. The drone velocity
        # rides along so a seed can turn relative motion back into the car's own.
        for measurement_type, stamp, z, R, cam, checkpoints in measurements:
            if not self._UKF_filter.update(measurement_type, stamp, z, R, now_sec, cam=cam,
                                           quad_vel=self.quad_vel):
                self.get_logger().warn(
                    f"UKF {measurement_type} update failed!",
                    throttle_duration_sec=2.0,
                )
            # One latency sample per new image (the AprilTag TF lookup returns the same
            # transform every tick until a new one arrives)
            if self.DIAGNOSTICS_ENABLED and stamp != self._latency_last_stamp[measurement_type]:
                self._latency_last_stamp[measurement_type] = stamp
                t_done = self.get_clock().now().nanoseconds * 1e-9
                self._record_latency(measurement_type, stamp, checkpoints, now_sec, t_done)

        # Divergence guard: the gate rejecting every frame means the filter has lost the
        # car (e.g. a bad seed). Re-seed rather than coast; the control loop aborts any
        # descent in progress and waits for the new seed before descending again.
        if self._UKF_filter.consecutive_rejections >= self.UKF_MAX_CONSEC_REJECT:
            self.get_logger().warn(
                f"UKF: {self._UKF_filter.consecutive_rejections} consecutive rejections - re-seeding"
            )
            self._UKF_filter.reset()
            self._UKF_filter.x[LP_State.PZ] = self.LANDING_HEIGHT_ABOVE_GND - self.quad_pose[QUAD_State.Z]
            self._ukf_diverged = True

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

        # Also compensate for the drag on the drone while it coasts level after the cut.
        # cD is a force coefficient (F = cD v|v|, as in the PID), so the deceleration is
        # (cD / m) v^2: v(t) = v0 / (1 + k v0 t), distance = ln(1 + k v0 t) / k, k = cD / m.
        quad_vel_eff = np.array(self.quad_vel, dtype=float)
        k = (self.PID_DRAG_COEFFICIENT / self.PID_MASS) \
            * np.hypot(quad_vel_eff[QUAD_State.X], quad_vel_eff[QUAD_State.Y]) * t
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

        # Relative velocity = pad velocity (axle velocity + rotation about the axle) - quad
        fwd_yaw = self._UKF_forward_predict_x[LP_State.YAW]
        self.landing_pad_yaw_forward_predict = fwd_yaw
        fwd_pv = self._pad_velocity(self._UKF_forward_predict_x)
        self.landing_pad_relative_velocity_forward_predict = np.array([
            fwd_pv[0] - self.quad_vel[0],
            fwd_pv[1] - self.quad_vel[1],
            -self.quad_vel[2],
        ])

        # Publish the actual landing pad pose estimate for all other uses
        # (diagnostics etc.)
        self.landing_pad_relative_position = np.array(
            [x[LP_State.PX], x[LP_State.PY], x[LP_State.PZ]]
        )

        yaw = x[LP_State.YAW]
        pv = self._pad_velocity(x)
        rel_vx = pv[0] - self.quad_vel[0]
        rel_vy = pv[1] - self.quad_vel[1]
        rel_vz = -self.quad_vel[2]
        self.landing_pad_relative_velocity = np.array([rel_vx, rel_vy, rel_vz])
        self.landing_pad_acceleration = self._pad_acceleration(x)

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


    def _record_latency(self, stream, stamp, checkpoints, t_tick, t_done) -> None:
        """ Turn one measurement's timestamps into per-stage latencies (ms) and queue
            them for the next CSV row.

            Detector stages are the gaps between consecutive checkpoints, starting from
            the image stamp. Then:
                wait  = last checkpoint -> UKF tick start (polling + transport)
                apply = UKF tick start  -> this update done
                total = image stamp     -> this update done (always available)
            Stages are NaN when the detector sent no (or non-finite) checkpoints.

        :param stream:      "apriltag" or "yolo"
        :param stamp:       Image timestamp (s)
        :param checkpoints: Detector end-of-stage timestamps (s), in LATENCY_STAGES
                            order, or None
        :param t_tick:      Start of the UKF tick that consumed the measurement (s)
        :param t_done:      Time the update call returned (s)
        """
        stages = LATENCY_STAGES[stream]
        lags = dict.fromkeys(stages + ("wait",), np.nan)

        if checkpoints is not None:
            cp = np.asarray(checkpoints, dtype=float)
            if cp.size == len(stages) and np.all(np.isfinite(cp)):
                prev = stamp
                for name, t in zip(stages, cp):
                    lags[name] = (t - prev) * 1000.0
                    prev = t
                lags["wait"] = (t_tick - prev) * 1000.0

        lags["apply"] = (t_done - t_tick) * 1000.0
        lags["total"] = (t_done - stamp) * 1000.0
        self._latency_pending[stream] = lags


    def _pad_velocity(self, state):
        """ Pad velocity in the level frame: rear-axle velocity along the heading plus
            the pad's rotation about the axle (omega x r). """
        yaw, om, v = state[LP_State.YAW], state[LP_State.YAW_RATE], state[LP_State.V]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = self._UKF_filter.axle_to_pad
        rx, ry = c * dx - s * dy, s * dx + c * dy
        return np.array([v * c - om * ry, v * s + om * rx])


    def _pad_acceleration(self, state):
        """ Pad acceleration in the level frame: rear-axle acceleration along the heading
            (a), centripetal acceleration of the axle (v * omega, to the left for omega > 0),
            and the pad's rotation about the axle (-omega^2 r). Yaw acceleration is not
            a state, so its (omega_dot x r) term is omitted. Derivative of _pad_velocity. """
        yaw, om = state[LP_State.YAW], state[LP_State.YAW_RATE]
        v, a = state[LP_State.V], state[LP_State.A]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = self._UKF_filter.axle_to_pad
        rx, ry = c * dx - s * dy, s * dx + c * dy
        return np.array([a * c - v * om * s - om**2 * rx,
                         a * s + v * om * c - om**2 * ry])


    # region ---- CTRL LOOP ----
    def _control_loop(self) -> None:
        """ Run main control and orchestration loop. Handles state machine logic.
        """

        # Get timestamp
        start = self.get_clock().now()
        t_now = start.nanoseconds * 1e-9

        # Reset logging
        self._pid_controller.reset_outputs()
        self._pad_acc_ff = np.full(2, np.nan)

        # Horizontal error to the forward-predicted pad
        err = np.hypot(*self.landing_pad_relative_position_forward_predict[:2]) #type: ignore

        # Consume the UKF divergence flag (set when the filter re-seeded itself)
        diverged, self._ukf_diverged = self._ukf_diverged, False

        # ---- State 0000 (Pre-arm)
        # Below conditions need to be met ALWAYS so we check regardless of state.
        # A link dropout mid-flight must not wipe the mission state: hold where we are
        # (MAVROS can't forward setpoints anyway) and resume on reconnection.
        if not self.fcu_state.connected:
            if not self._armed_confirmed:
                self.controller_state = 0
            return

        if not self._mode_confirmed:
            if not self._mode_requested and not self.WAIT_FOR_PILOT_GUIDED:
                self._request_mode()
            elif self.WAIT_FOR_PILOT_GUIDED:
                self.get_logger().info(
                    f"Waiting for the pilot to select {self._mode} "
                    f"(FCU in {self.fcu_state.mode or 'unknown'})...",
                    throttle_duration_sec=5.0,
                )
            self.controller_state = 0
            return

        # Left GUIDED again before arming (e.g. a pre-flight mode-switch check): just go
        # back to waiting for GUIDED, nothing has started yet
        if self.fcu_state.mode != self._mode and not self._armed_confirmed:
            self._mode_confirmed = False
            self._mode_requested = False
            self.controller_state = 0
            return

        # ---- State 8000 (Pilot override - passive)
        # Once armed, the FCU leaving GUIDED without us requesting RTL means the pilot (or
        # an FCU failsafe) has the vehicle. Stop commanding it for good, but keep logging.
        if not self._rtl_initiated and (self._pilot_override or self.fcu_state.mode != self._mode):
            if not self._pilot_override:
                self._pilot_override = True
                self.get_logger().warn(
                    f"FCU left {self._mode} (now {self.fcu_state.mode}) - pilot override, "
                    "controller passive for the rest of this run"
                )
            self.controller_state = 8000
            if self.DIAGNOSTICS_ENABLED:
                self.log_diagnostics()
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
                d_wait = np.hypot(d_north, d_east)

                # A waiting point outside the boundary box is a config error (e.g. the
                # SITL coordinates left in the launch file): refuse to fly there.
                if d_wait > self.BOUNDARY_LIMIT:
                    self.get_logger().error(
                        f"GPS waiting point is {d_wait:.0f} m away (> boundary_limit "
                        f"{self.BOUNDARY_LIMIT:.0f} m) - check gps_lat / gps_lon"
                    )
                    self.controller_state = 7000
                elif d_wait <= self.GPS_LOC_BUFFER:
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
        # Also hold until the car is in front of the drone (relative bearing within
        # +-LANDING_ACQUIRE_BEARING_MAX of its heading), so the chase never starts by
        # accelerating backwards and then forwards again. Needs a seeded UKF: before that
        # the relative position is a placeholder (0, 0).
        rel = self.landing_pad_relative_position_forward_predict
        pad_in_front = (
            self._UKF_start
            and not self._UKF_filter.seed_pending
            and abs(self._UKF_filter._wrap(np.arctan2(rel[1], rel[0]) - self.quad_yaw))
            < self.LANDING_ACQUIRE_BEARING_MAX
        )
        if (
            self.controller_state == 2100
            and self._landing_pad_found
            and t_now - self._landing_pad_first_seen_time > self.LANDING_TIME_VISUAL_TIME_SP
            and pad_in_front
        ):
            self.get_logger().info(
                f"Landing Pad visual hold ok, starting {self.LANDING_PAD_LOCK_TIME_SP} sec timer..."
            )
            self._lock_timer_start = t_now
            self.controller_state = 3000

        # ---- State 3000 (Move over and Maintain Landing Pad Lock)
        # Descend once centred, either early when the estimate is confident (for
        # LANDING_CONFIDENT_DWELL) or, as a fallback, once the lock timer runs out.
        # Never while the UKF is (re-)seeding: its state is a placeholder until then.
        if self.controller_state == 3000:
            d = self._UKF_diag
            pad_speed = abs(self._UKF_filter.x[LP_State.V])
            pos_sigma = np.hypot(d["sigma_px"], d["sigma_py"])
            # Pad velocity = axle speed along heading + rotation about the axle, so heading
            # uncertainty adds v*sigma_yaw sideways and yaw-rate uncertainty |r|*sigma_omega
            arm = float(np.hypot(*self._UKF_filter.axle_to_pad))
            vel_sigma = float(np.sqrt(d["sigma_v"]**2 + (pad_speed * d["sigma_yaw"])**2
                                      + (arm * d["sigma_yaw_rate"])**2))
            # Current (not forward-predicted) relative speed: the forward one compares the
            # pad's velocity T s ahead with the drone's now, so it never reaches zero in a turn
            rel_speed = np.hypot(*self.landing_pad_relative_velocity[:2]) #type: ignore
            centred = self._landing_pad_found and err < self.LANDING_CENTERED_ERROR_THRESHOLD

            confident_now = (
                centred
                and self._UKF_start
                and not self._UKF_filter.seed_pending
                and pos_sigma < self.LANDING_CONFIDENT_POS_SIGMA
                and vel_sigma < self.LANDING_CONFIDENT_VEL_SIGMA
                and rel_speed < self.LANDING_CONFIDENT_REL_VEL
            )
            self._confident_since = (self._confident_since or t_now) if confident_now else None

            confident = (
                self._confident_since is not None
                and t_now - self._confident_since >= self.LANDING_CONFIDENT_DWELL
            )
            self._gate_pos_sigma, self._gate_vel_sigma = pos_sigma, vel_sigma
            self._gate_rel_speed, self._gate_confident = rel_speed, int(confident)
            if self._lock_timer_start is None:
                self._lock_timer_start = t_now
            timed_out = t_now - self._lock_timer_start > self.LANDING_PAD_LOCK_TIME_SP

            if centred and not self._UKF_filter.seed_pending and (confident or timed_out):
                self.get_logger().info(
                    f"Landing Pad Acquired ({'confident' if confident else 'lock timer'}) - "
                    f"{t_now - self._lock_timer_start:.1f} s after lock started, "
                    f"err={err:.2f} m, sigma_pos={pos_sigma:.2f} m, "
                    f"sigma_vel={vel_sigma:.2f} m/s, rel_vel={rel_speed:.2f} m/s"
                )
                self.controller_state = 4000
        else:
            self._confident_since = None
            self._gate_pos_sigma = self._gate_vel_sigma = self._gate_rel_speed = np.nan
            self._gate_confident = 0

        # ---- State 4000 (Begin Landing Descent)
        if self.controller_state == 4000:
            self.target_alt = 0.0 # self.LANDING_HEIGHT_THRESHOLD  # land
            h = -self.landing_pad_relative_position[LP_State.PZ]

            # A re-seeded UKF has no usable estimate: climb out and retry rather than
            # descend on a placeholder state.
            if diverged:
                self.get_logger().warn("Landing Aborted - UKF re-seeding")
                self._landing_attempts += 1
                self.controller_state = 5000
            # Fallback touchdown detection: prioritize the measured velocity impact
            # over the height-based cutoff, which can fail if the height estimate drifts.
            elif self._z_touchdown_trigger:
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
                    # Retry: restart the lock timer so the descent waits for confidence again
                    # (the timer remains the fallback if confidence never comes)
                    self._lock_timer_start = t_now
                    self.get_logger().info(
                        f"Recovered after abort, retrying - starting {self.LANDING_PAD_LOCK_TIME_SP} sec timer..."
                    )
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
            # Pad acceleration feedforward. Its direction comes from the yaw estimate, so it
            # is only applied once the heading is trusted (same gate as marker_yaw).
            if marker_yaw is not None:
                self._pad_acc_ff = self.PID_ACCEL_FF_GAIN * self.landing_pad_acceleration
            else:
                self._pad_acc_ff = np.zeros(2)
            # Position target = forecast cut point (forward-predicted). Velocity = CURRENT
            # relative velocity: that is the rate of change of the forecast error in steady
            # tracking, whereas the forward-predicted velocity carries a T * a_pad bias
            # (inward in turns) that the D term would act on.
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
                landing_pad_relative_velocity=self.landing_pad_relative_velocity,
                landing_pad_acceleration=self._pad_acc_ff,
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
        if self._rtl_initiated or self._pilot_override or not self._tko_reached or self._disarmed:
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
        """ Buffer AprilTag pipeline timestamps, keyed by image stamp, until the UKF
            loop picks up the matching transform. (YOLO carries its timestamps inside
            the measurement instead.)
        """
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._pipeline_timing_buffer[key] = msg
        if len(self._pipeline_timing_buffer) > 50:
            self._pipeline_timing_buffer.pop(next(iter(self._pipeline_timing_buffer)))


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

        # AprilTag pipeline timestamps (side topic). Reliable with some depth, so a busy
        # executor doesn't silently drop the timing for a measurement it then consumes.
        _timing_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pipeline_timing_sub = self.create_subscription(
            Vector3Stamped,
            "/landing_pad/pipeline_timing",
            self._pipeline_timing_callback,
            _timing_qos
        )
        self._pipeline_timing_buffer: dict[tuple[int, int], Vector3Stamped] = {}

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

                # Landing pad global data (drone reference = ground truth in SITL, FCU
                # estimate when ground_truth_available is False)
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

                # Latency through pipeline (ms), one sample per measurement: filled on
                # the first row after a new measurement is applied, NaN on all other
                # rows. apriltag_/yolo_ {detector stages..., wait, apply, total}, see
                # LATENCY_STAGES and _record_latency.
                *LATENCY_COLUMNS,

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

                # Filtered position at the latest YOLO image stamp (NaN if rejected)
                "yolo_post_rel_x", "yolo_post_rel_y", "yolo_post_rel_z",

                # Early-acquisition (confidence) gate inputs, state 3000 only
                "gate_pos_sigma", "gate_vel_sigma", "gate_rel_speed", "gate_confident",

                # Box-derived yaw fed to the UKF (global/local frame, rad) and its 1-sigma,
                # NaN when not used on that frame
                "landing_pad_yolo_car_glob_yaw_raw", "landing_pad_yolo_car_yaw_sigma_raw",

                # Pad acceleration feedforward applied to the PD (m/s^2, level frame),
                # NaN when the PD isn't running, 0 while the heading isn't trusted
                "pad_acc_ff_x", "pad_acc_ff_y",

                # Drone vertical acceleration (m/s^2, up +), differentiated from the
                # odometry vz exactly as the touchdown fallback sees it
                "quad_z_accel",
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

        # Pipeline latency: popped, so each measurement's sample lands on exactly one row
        latency = []
        for stream, stages in LATENCY_STAGES.items():
            lags = self._latency_pending.pop(stream, {})
            latency += [lags.get(k, np.nan) for k in stages + LATENCY_TAIL]

        # YOLO raw is a fitted position (fallback) or box edges in pixels (box mode)
        if yolo["box"]:
            shadow_ok = (self._yolo_shadow_fit is not None
                         and self._yolo_shadow_stamp == yolo["raw_stamp"])
            yolo_pos = self._yolo_shadow_fit if shadow_ok else np.full(3, np.nan)
        else:
            yolo_pos = yolo["raw"][:3]
        yolo_box = yolo["raw"] if yolo["box"] else np.full(4, np.nan)

        # Box-derived yaw actually fed to the UKF on this frame (NaN when the yaw slot was
        # unused: box mode, seeding, gates closed or hypotheses unresolved). Stamp-matched,
        # so it is only logged against its own image.
        fy = self._yolo_fit_yaw
        if fy is not None and fy[0] == yolo["raw_stamp"]:
            yolo_fit_yaw, yolo_fit_yaw_sigma = fy[1], fy[2]
        else:
            yolo_fit_yaw = yolo_fit_yaw_sigma = np.nan

        # Filtered position at the YOLO image stamp (needs the matching ukf.py change)
        yolo_post = yolo.get("post_x", np.full(3, np.nan))

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
            *yolo_post,
            self._gate_pos_sigma, self._gate_vel_sigma,
            self._gate_rel_speed, self._gate_confident,
            yolo_fit_yaw, yolo_fit_yaw_sigma,
            *self._pad_acc_ff,
            self.quad_z_accel,
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
            c, s = np.cos(pad_true_yaw), np.sin(pad_true_yaw)
            dx, dy = self.YOLO_CAR_PAD_POSITION[0], self.YOLO_CAR_PAD_POSITION[1]
            pad_true_xy = p_pos[:2] + np.array([c * dx - s * dy, s * dx + c * dy])
            e = np.array([*(pad_true_xy - q_pos[:2]), pad_true_yaw]) - self._UKF_filter.x[nees_states]
            e[-1] = self._UKF_filter._wrap(e[-1])
            try:
                nees = float(e @ np.linalg.solve(Pc[np.ix_(nees_states, nees_states)], e))
            except np.linalg.LinAlgError:
                nees = np.nan
        else:
            q_pos = q_vel = p_pos = p_vel = np.full(3, np.nan)
            quad_true_yaw = pad_true_yaw = nees = np.nan

        # Reference drone pose for the *_glob_* columns: ground truth in SITL, the FCU's
        # own estimate in real flight (so the global pad track is still logged)
        q_ref_pos = q_pos if self.GROUND_TRUTH_AVAILABLE else self.quad_pose
        q_ref_vel = q_vel if self.GROUND_TRUTH_AVAILABLE else self.quad_vel

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
                yolo["raw_stamp"], *yolo_pos, int(yolo["box"]), *yolo_box, #type: ignore
                *est_pos, *est_vel, self.landing_pad_yaw - quad_true_yaw,
                *(p_pos - q_pos), *(p_vel - q_vel), pad_true_yaw - quad_true_yaw,

                # Landing pad global data
                *(apr["raw"][:3] + q_ref_pos), apr["raw"][LP_Meas.YAW],
                *(yolo_pos + q_ref_pos), #type: ignore
                *(est_pos + q_ref_pos), *(est_vel + q_ref_vel),
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

                # Latency through pipeline (same order as LATENCY_COLUMNS)
                *latency,
            ] + extra
        )

        # Flush about once a second so a flat battery or a killed process costs at most
        # ~1 s of log, not the whole write buffer
        if current_time - self._csv_last_flush >= 1.0:
            self._csv_file.flush()
            self._csv_last_flush = current_time


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