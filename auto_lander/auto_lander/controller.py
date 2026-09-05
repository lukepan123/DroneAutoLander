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
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandTOL
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import MessageInterval
from datetime import datetime

from .state_definitions import QUAD_State
from .state_definitions import LP_State
from .ukf import UKF
from .mpc_controller import MPCController
from .pid_controller import PIDController

""" Main orchestrator control loop, handles high level controller state logic and 
    orchestration of UKF and low-level controller for autonomous landing process. 

    Landing algorithim consists of several states.
"""

class Orchestrator(Node):
    """ Defines the orchestrator node."""

    def __init__(self) -> None:
        super().__init__("orchestrator")
        # ---- NODE PARAMETERS ----
        # Enable logging and certain diagnostics
        self.declare_parameter("diagnostics_enabled", True)
        self.diagnostics_enabled = (
            self.get_parameter("diagnostics_enabled").get_parameter_value().bool_value
        )
        # If in sim, log ground truth
        self.declare_parameter("ground_truth_available", True)
        self.ground_truth_available = (
            self.get_parameter("ground_truth_available").get_parameter_value().bool_value
        )


        # ---- Global State Variables ----
        self.MAX_RUNTIME = 200.0  # secs
        self.BOUNDARY_LIMIT = 500.0  # m x m square
        self.LANDING_HEIGHT_THRESHOLD = 0.3  # m above landing pad
        self.LANDING_ERROR_THRESHOLD = 0.1  # m error

        self.target_z = 2.0  # m

        self.controller_state = 0
        self.fcu_state = State()
        self.odometry  = Odometry()
        self.quad_pose = np.zeros(3)
        self.quad_vel  = np.zeros(3)
        self.quad_roll  = 0.0
        self.quad_pitch = 0.0
        self.quad_yaw   = 0.0

        self.landing_pad_relative_odometry = Odometry()
        self.landing_pad_relative_position = np.zeros(3)
        self.landing_pad_relative_position_forward_predict = np.zeros(3)
        self.landing_pad_relative_velocity = np.zeros(3)
        self.landing_pad_relative_velocity_forward_predict = np.zeros(3)
        self.landing_pad_yaw = 0.0
        self.landing_pad_yaw_forward_predict = 0.0

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
        self._tko_altitude_SP = self.target_z  # Inititalise at initial target altitude

        # ---- State 2xxx (Searching for Landing Pad) Variables ----
        self._landing_pad_found = False
        self._landing_pad_first_seen_time = None
        self._landing_pad_visual_time_SP = 2.0

        # ---- State 3xxx (Maintaining Landing Pad Lock) Variables ----
        self._landing_pad_locked_time_SP = self._landing_pad_visual_time_SP + 20.0
        self._landing_pad_lost_time = None
        self._landing_pad_lost_time_SP = 2.0

        # ---- State 4xxx (Beginning Landing Descent) Variables ----
        self._landed_time = None
        self.cutoff = False

        # ---- State 5xxx (Landing Aborted) Variables ----
        self._landing_attempts = 0

        # ---- State 6xxx (Landing Confirmed) Variables ----
        self._idle_before_RTL_SP = 5.0

        # ---- State 7xxx (RTL) Variables ----
        self._rtl_initiated = False

        # ---- CONTROL CLASS INITIALISATIONS ----
        self._UKF_start = False
        self._UKF_last_update = self.get_clock().now()
        self._UKF_timer_rate = 0.05
        self._UKF_filter = UKF()
        self._UKF_forward_predict_x = None

        self._UKF_diag = self._UKF_filter.get_covar_diagnostics()

        self._UKF_last_apriltag_stamp = Time().to_msg()
        self._UKF_raw_measurement = [0.0, 0.0, 0.0, 0.0]
        self._UKF_raw_measurement_stamp = 0.0

        self._UKF_yolo_raw_measurement = [0.0, 0.0, 0.0, 0.0]
        self._UKF_yolo_raw_measurement_stamp = 0.0
        self._UKF_last_yolo_stamp = Time().to_msg()

        self._cam_to_image_lag = None
        self._image_to_transform_lag = None
        self._transform_to_UKF_lag = None
        self._yolo_cam_to_image_lag = None
        self._yolo_image_to_transform_lag = None
        self._yolo_transform_to_UKF_lag = None
        self._UKF_meas_age = 0.0

        self._control_timer_rate = 0.1
        # self._mpc_controller = MPCController()
        self._pid_controller = PIDController(self._control_timer_rate)

        # ---- SUBSCRIPTIONS ----
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
        self._twist_sub = self.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local", #TODO: Check on real drone if local
            self._vel_accel_callback,
            _odom_qos,
        )
        self._landing_pad_found_sub = self.create_subscription(
            Bool, "/landing_pad/found", self._landing_pad_found_callback, 1
        )

        # ---- PUBLISHERS ----
        self.att_pub = self.create_publisher(
            AttitudeTarget, "/mavros/setpoint_raw/attitude", 10
        )
        self.vel_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10
        )

        # ---- TF2 ----
        self._tf_map_landing_pad_buffer = tf2_ros.Buffer(
            node=self, cache_time=Duration(seconds=10)
        )
        self._tf_map_landing_pad_listener = tf2_ros.TransformListener(
            self._tf_map_landing_pad_buffer, self
        )

        # ---- CLIENTS ----
        self._set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self._arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self._takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self._message_interval_client = self.create_client(
            MessageInterval, "/mavros/set_message_interval"
        )

        # ---- CONTROL TIMERS ----
        self._UKF_timer = self.create_timer(self._UKF_timer_rate, self._UKF_loop)
        self._safety_timer_rate = 1.0
        self._safety_timer = self.create_timer(
            self._safety_timer_rate, self._safety_loop
        )
        self._control_timer = self.create_timer(
            self._control_timer_rate, self._control_loop
        )

        self._set_message_intervals()  # MAVROS message rates

        #  ---- DIAGNOSTICS AND LOGGING ----
        """ Generate a log if diagnostics and logging enabled."""
        self.quad_true_odometry = Odometry()
        self.landing_pad_true_odometry = Odometry()
        if self.diagnostics_enabled:
            self.start_diagnostics()
        
        self.get_logger().info("Auto Lander (callbacks) started")


    # ---- CALLBACK IMPLEMENTATIONS ----
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

        self.quad_vel = np.array(
            [
                self.odometry.twist.twist.linear.x,
                self.odometry.twist.twist.linear.y,
                self.odometry.twist.twist.linear.z,
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


    def _vel_accel_callback(self, msg: TwistStamped) -> None:
        """Obtain FCU linear and angular velocities rates and derive accelerations.

        :param msg: TwistStamped msg from quadcopter
        """

        self.odometry.twist.twist.angular.x = msg.twist.angular.x
        self.odometry.twist.twist.angular.y = msg.twist.angular.y
        self.odometry.twist.twist.angular.z = msg.twist.angular.z


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

    def _UKF_loop(self):
        """ Run UKF predict and update loops. Manages state estimation of the landing
            pad platform. Calls upon both the apriltag and YOLO measurements
        """

        # Latch start of UKF for landing pad on aquisition of landing pad from detector 
        # node, if its been too long since we've seen the landing pad, reset the UKF
        if self._landing_pad_found and not self._UKF_start:
            self._UKF_filter.reset()
            self._UKF_start = True

        # Get timestamp from last cycle
        now = self.get_clock().now()
        dt = (now - self._UKF_last_update).nanoseconds * 1e-9
        self._UKF_last_update = now

        # Predict UKF step (after first measurement)
        if self._UKF_start:
            self._UKF_filter.predict(
                self.quad_vel, 
                dt, 
                now.nanoseconds * 1e-9
            )

        # Warn in logs if filter goes non-PD or any sigma blows up
        if not self._UKF_diag["is_pd"]:
            self.get_logger().warn(
                "UKF: covariance matrix is no longer positive definite!"
            )

        # Update AprilTag UKF step
        try:
            tf_msg = self._tf_map_landing_pad_buffer.lookup_transform(
                "local", "landing_pad_link", Time()
            )
            stamp_sec = Time.from_msg(tf_msg.header.stamp).nanoseconds / 1e9
            stamp_is_new = (
                tf_msg.header.stamp.sec != self._UKF_last_apriltag_stamp.sec
                or tf_msg.header.stamp.nanosec != self._UKF_last_apriltag_stamp.nanosec
            )

            if stamp_is_new:
                # Measurement age — how stale is this TF?
                self._UKF_meas_age = (
                    now - Time.from_msg(tf_msg.header.stamp)
                ).nanoseconds / 1e6

                # Extract pose measurement
                t = np.array([
                    tf_msg.transform.translation.x,
                    tf_msg.transform.translation.y,
                    tf_msg.transform.translation.z,
                ])

                q = tf_msg.transform.rotation
                _, _, yaw = tf_transformations.euler_from_quaternion(
                    [q.x, q.y, q.z, q.w]
                )
                measurement = np.array([t[0], t[1], t[2], yaw])

                # Store raw measurement
                self._UKF_raw_measurement = [t[0], t[1], t[2], yaw]
                self._UKF_raw_measurement_stamp = stamp_sec

                # Update measurement noise based on drone angular rates
                cov_adj_x = (1 + 2 * np.sqrt(
                        self.odometry.twist.twist.angular.y**2
                        + self.odometry.twist.twist.angular.z**2
                    )
                )
                cov_adj_y = (1 + 2 * np.sqrt(
                        self.odometry.twist.twist.angular.x**2
                        + self.odometry.twist.twist.angular.z**2
                    )
                )
                cov_adj_z = (1 + 2 * np.sqrt(
                        self.odometry.twist.twist.angular.x**2
                        + self.odometry.twist.twist.angular.y**2
                    )
                )

                self._UKF_filter.R = np.diag(
                    [
                        0.00009 * cov_adj_x,
                        0.00009 * cov_adj_y,
                        0.00009 * cov_adj_z,
                        0.002 * cov_adj_z,
                    ]
                )

                accepted = self._UKF_filter.update(measurement, stamp_sec)
                if not accepted:
                    self.get_logger().warn(
                        f"UKF last update failed!"
                    )

                # Log diagnostics if enabled
                if self.diagnostics_enabled:
                    key = (tf_msg.header.stamp.sec, tf_msg.header.stamp.nanosec)
                    pt = self._pipeline_timing_buffer.pop(key, None)
                    if pt is not None:
                        t0 = stamp_sec
                        t1 = pt.vector.x
                        t2 = pt.vector.y
                        now_sec = now.nanoseconds / 1e9
                        self._cam_to_image_lag = (t1 - t0) * 1000.0
                        self._image_to_transform_lag = (t2 - t1) * 1000.0
                        self._transform_to_UKF_lag = (now_sec - t2) * 1000.0
                    else:
                        self.get_logger().warn(
                            "No matching pipeline_timing message for this TF — lag breakdown skipped",
                            throttle_duration_sec=2.0,
                        )

            self._UKF_last_apriltag_stamp = tf_msg.header.stamp

        except Exception:
            pass  # Predict-only cycle, no correction this tick

        # Update YOLO UKF step
        try:
            tf_msg = self._tf_map_landing_pad_buffer.lookup_transform(
                "local", "landing_pad_link_yolo", Time()
            )
            stamp_sec = Time.from_msg(tf_msg.header.stamp).nanoseconds / 1e9
            stamp_is_new = (
                tf_msg.header.stamp.sec != self._UKF_last_yolo_stamp.sec
                or tf_msg.header.stamp.nanosec != self._UKF_last_yolo_stamp.nanosec
            )

            if stamp_is_new:
                # Measurement age — how stale is this TF?
                self._UKF_meas_age = (
                    now - Time.from_msg(tf_msg.header.stamp)
                ).nanoseconds / 1e6

                # Extract pose measurement
                t = np.array([
                    tf_msg.transform.translation.x,
                    tf_msg.transform.translation.y,
                    tf_msg.transform.translation.z,
                ])

                q = tf_msg.transform.rotation
                _, _, yaw = tf_transformations.euler_from_quaternion(
                    [q.x, q.y, q.z, q.w]
                )
                measurement = np.array([t[0], t[1], t[2], yaw])

                # Store raw measurement
                self._UKF_yolo_raw_measurement = [t[0], t[1], t[2], yaw]
                self._UKF_yolo_raw_measurement_stamp = stamp_sec

                # Update measurement noise
                self._UKF_filter.R = np.diag(
                    [
                        0.010,
                        0.010,
                        0.050,
                        1e6, # We get no yaw information
                    ]
                )

                accepted = self._UKF_filter.update(measurement, stamp_sec)
                if not accepted:
                    self.get_logger().warn(
                        f"UKF last update failed!"
                    )

                # Log diagnostics if enabled
                if self.diagnostics_enabled:
                    key = (tf_msg.header.stamp.sec, tf_msg.header.stamp.nanosec)
                    pt = self._yolo_pipeline_timing_buffer.pop(key, None)
                    if pt is not None:
                        t0 = stamp_sec
                        t1 = pt.vector.x
                        t2 = pt.vector.y
                        now_sec = now.nanoseconds / 1e9
                        self._yolo_cam_to_image_lag = (t1 - t0) * 1000.0
                        self._yolo_image_to_transform_lag = (t2 - t1) * 1000.0
                        self._yolo_transform_to_UKF_lag = (now_sec - t2) * 1000.0
                    else:
                        self.get_logger().warn(
                            "No matching pipeline_timing message for this TF — lag breakdown skipped",
                            throttle_duration_sec=2.0,
                        )

            self._UKF_last_yolo_stamp = tf_msg.header.stamp

        except Exception:
            pass  # Predict-only cycle, no correction this tick

        # ---- Grab per-state covariance diagnostics every tick ----
        self._UKF_diag = self._UKF_filter.get_covar_diagnostics()

        # Always publish current UKF state
        x = self._UKF_filter.x

        # Forward predict by the time it would take for the drone to drop from its 
        # altitude to the landing pad. This is the value used by the controller
        # Added a fudge factor...
        t = np.sqrt(2 * 9.81 * self.LANDING_HEIGHT_THRESHOLD) / 9.81
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
        rel_vx = self._UKF_forward_predict_x[LP_State.V] * np.cos(self.landing_pad_yaw_forward_predict) - self.quad_vel[0]
        rel_vy = self._UKF_forward_predict_x[LP_State.V] * np.sin(self.landing_pad_yaw_forward_predict) - self.quad_vel[1]
        rel_vz = -self.quad_vel[2]
        self.landing_pad_relative_velocity_forward_predict = np.array([rel_vx, rel_vy, rel_vz])
        
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
        if elapsed > self._UKF_timer_rate * 1000:
            self.get_logger().warn(f"UKF loop took {elapsed:.2f} ms!")


    def _control_loop(self) -> None:
        """ Run main control and orchestration loop. Handles state machine logic.
        """

        # Get timestamp
        start = self.get_clock().now()

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
            self.controller_state = 2000

        # ---- State 2000 (Searching for Landing Pad)
        if self.controller_state == 2000:
            if self._landing_pad_found and self._landing_pad_first_seen_time is None:
                self._landing_pad_first_seen_time = (
                    self.get_clock().now().nanoseconds / 1e9
                )
                self.get_logger().info(
                    f"Landing Pad found, starting {self._landing_pad_visual_time_SP} sec timer..."
                )
                self.controller_state = 2100

        # ---- State 2100 (Maintain Landing Pad Visual Lock - Yaw to match target)
        if self.controller_state == 2100:
            now = self.get_clock().now().nanoseconds / 1e9

            if self._landing_pad_found:
                self._landing_pad_lost_time = None
                if (now - self._landing_pad_first_seen_time) > self._landing_pad_visual_time_SP:
                    self.get_logger().info(
                        f"Landing Pad visual hold ok, starting {self._landing_pad_locked_time_SP} sec timer..."
                    )
                    self.controller_state = 3000
            else:
                if self._landing_pad_lost_time is None:
                    self._landing_pad_lost_time = now
                else:
                    if (now - self._landing_pad_lost_time) > self._landing_pad_lost_time_SP:
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
                ) > self._landing_pad_locked_time_SP:
                    self.get_logger().info("Landing Pad Acquired")
                    self.controller_state = 4000
            else:
                if self._landing_pad_lost_time is None:
                    self._landing_pad_lost_time = now
                else:
                    if (
                        now - self._landing_pad_lost_time
                    ) > self._landing_pad_lost_time_SP:
                        self._landing_pad_first_seen_time = None
                        self._landing_pad_lost_time = None
                        self.get_logger().info("Landing Pad Lost!")
                        self.controller_state = 2000

        # ---- State 4000 (Begin Landing Descent)
        if self.controller_state == 4000:
            now = self.get_clock().now().nanoseconds / 1e9
            self.target_z = self.LANDING_HEIGHT_THRESHOLD  # land

            # Check we still have the target in view, reset timer if so
            if self._landing_pad_found:
                self._landing_pad_lost_time = None
            else:
                if self._landing_pad_lost_time is None:
                    self._landing_pad_lost_time = now
                else:
                    if (
                        now - self._landing_pad_lost_time
                    ) > self._landing_pad_lost_time_SP:
                        self._landing_pad_first_seen_time = None
                        self._landing_pad_lost_time = None
                        self.get_logger().info("Landing Pad Lost!")
                        self.controller_state = 2000

            # Check target error is not bad compared to where we want to be
            err_x = abs(self.landing_pad_relative_position_forward_predict[0])
            err_y = abs(self.landing_pad_relative_position_forward_predict[1])

            if (
                self.quad_pose[QUAD_State.Z] <= 0.6
                and err_x < self.LANDING_ERROR_THRESHOLD
                and err_y < self.LANDING_ERROR_THRESHOLD
            ):
                self.cutoff = True
                self._landed_time = now
                self.get_logger().info(f"Throttle Cut Engaged - Err_x = {err_x}, Err_y = {err_y}")
                self.controller_state = 6000
            elif self.quad_pose[QUAD_State.Z] <= 0.6 and (
                err_x >= self.LANDING_ERROR_THRESHOLD 
                or err_y >= self.LANDING_ERROR_THRESHOLD
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
                        ) > self._landing_pad_lost_time_SP:
                            self._landing_pad_first_seen_time = None
                            self._landing_pad_lost_time = None
                            self.get_logger().info("Landing Pad Lost!")
                            self.controller_state = 2000
                
                # Regain altitude, then try again
                self.target_z = 3.0
                if self.quad_pose[QUAD_State.Z] > 2.5:
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

        # ---- Run Controller ----
        if self.controller_state >= 2000 and self.controller_state < 3000:
            self._pid_controller.stop(self) # Sit still and observe
        elif self.controller_state >= 3000 and self.controller_state <= 6000:
            # self._mpc_controller.compute_control(self)
            self._pid_controller.update(self) # Chase

        # Log diagnostics
        if self.diagnostics_enabled:
            self.log_diagnostics()

        # Check for timer overruns
        end = self.get_clock().now()
        elapsed = (end - start).nanoseconds / 1e6  # ms
        if elapsed > self._control_timer_rate * 1000:
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
        if (self.controller_state < 6000 and self._UKF_diag["covar_max_eig"] > 20.0):
            self.get_logger().warn(
                f"UKF: max eigenvalue {self._UKF_diag['covar_max_eig']:.3f} — filter diverging - aborting",
                throttle_duration_sec=1.0
            )
            self._initiate_rtl("Landing Estimate too Bad")

    
    # ---- DIAGNOSTICS FUNCTION CALLBACKS ----
    def _landing_pad_true_odometry_callback(self, msg: Odometry):
        """ SITL ONLY: Pass true odometry of landing pad for data.
        """

        self.landing_pad_true_odometry = msg


    def _true_odometry_callback(self, msg: Odometry):
        """ SITL ONLY: Pass true (ground truth) odometry of the quad itself for data 
            logging.
        """

        self.quad_true_odometry = msg


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
        if self.ground_truth_available:
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
                "landing_pad_raw_stamp",
                "landing_pad_rel_x_raw",
                "landing_pad_rel_y_raw",
                "landing_pad_rel_z_raw",
                "landing_pad_rel_yaw_raw",
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
                "landing_pad_glob_x_raw",
                "landing_pad_glob_y_raw",
                "landing_pad_glob_z_raw",
                "landing_pad_glob_yaw_raw",
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
                "nees",

                # Latency through pipeline
                "cam_to_image_lag",
                "image_to_transform_lag",
                "transform_to_UKF_lag",
                "yolo_cam_to_image_lag",
                "yolo_image_to_transform_lag",
                "yolo_transform_to_UKF_lag",
                "total_lag",
            ]
        )
        self.get_logger().info(f"CSV logging initialized: {self._csv_filename}")


    def log_diagnostics(self):
        """ Log data from controller into .csv file. 
        """

        # UKF Diagnostics
        d = self._UKF_diag  # shorthand

        # For live plotting publishing
        self.landing_pad_relative_odometry_pub.publish(self.landing_pad_relative_odometry)

        UKF_raw_msg = Vector3Stamped()
        UKF_raw_msg.header.stamp = self.landing_pad_relative_odometry.header.stamp # Compare at same timestamp
        UKF_raw_msg.header.frame_id = "local"
        UKF_raw_msg.vector.x = self._UKF_raw_measurement[QUAD_State.X]
        UKF_raw_msg.vector.y = self._UKF_raw_measurement[QUAD_State.Y]
        UKF_raw_msg.vector.z = self._UKF_raw_measurement[QUAD_State.Z]
        self.landing_pad_relative_raw_pub.publish(UKF_raw_msg)

        current_time = self.get_clock().now().nanoseconds / 1e9

        # Only if ground-truth is available do we calculate the NEES, otherwise just 
        # log 0
        if self.ground_truth_available:
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
                    self._UKF_raw_measurement_stamp,
                    self._UKF_raw_measurement[QUAD_State.X],
                    self._UKF_raw_measurement[QUAD_State.Y],
                    self._UKF_raw_measurement[QUAD_State.Z],
                    self._UKF_raw_measurement[QUAD_State.YAW] - true_yaw,
                    self.landing_pad_relative_odometry.pose.pose.position.x,
                    self.landing_pad_relative_odometry.pose.pose.position.y,
                    self.landing_pad_relative_odometry.pose.pose.position.z,
                    self.landing_pad_relative_odometry.twist.twist.linear.x,
                    self.landing_pad_relative_odometry.twist.twist.linear.y,
                    self.landing_pad_relative_odometry.twist.twist.linear.z,
                    self.landing_pad_yaw - true_yaw,
                    self.landing_pad_true_odometry.pose.pose.position.x - self.quad_true_odometry.pose.pose.position.x,
                    self.landing_pad_true_odometry.pose.pose.position.y - self.quad_true_odometry.pose.pose.position.y,
                    self.landing_pad_true_odometry.pose.pose.position.z - self.quad_true_odometry.pose.pose.position.z,
                    # Fix velocity because it is given in body frame not world frame...
                    self.landing_pad_true_odometry.twist.twist.linear.x * np.cos(lp_pad_true_yaw) - self.quad_true_odometry.twist.twist.linear.x * np.cos(true_yaw),
                    self.landing_pad_true_odometry.twist.twist.linear.x * np.sin(lp_pad_true_yaw) - self.quad_true_odometry.twist.twist.linear.x * np.sin(true_yaw),
                    self.landing_pad_true_odometry.twist.twist.linear.z - self.quad_true_odometry.twist.twist.linear.z,
                    lp_pad_true_yaw - true_yaw,

                    # Landing pad global data
                    self._UKF_raw_measurement[QUAD_State.X] + self.quad_true_odometry.pose.pose.position.x,
                    self._UKF_raw_measurement[QUAD_State.Y] + self.quad_true_odometry.pose.pose.position.y,
                    self._UKF_raw_measurement[QUAD_State.Z] + self.quad_true_odometry.pose.pose.position.z,
                    self._UKF_raw_measurement[QUAD_State.YAW],
                    self.landing_pad_relative_odometry.pose.pose.position.x + self.quad_true_odometry.pose.pose.position.x,
                    self.landing_pad_relative_odometry.pose.pose.position.y + self.quad_true_odometry.pose.pose.position.y,
                    self.landing_pad_relative_odometry.pose.pose.position.z + self.quad_true_odometry.pose.pose.position.z,
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
                    nees,

                    # Latency through pipeline
                    self._cam_to_image_lag,
                    self._image_to_transform_lag,
                    self._transform_to_UKF_lag,
                    self._yolo_cam_to_image_lag,
                    self._yolo_image_to_transform_lag,
                    self._yolo_transform_to_UKF_lag,
                    self._UKF_meas_age,
                ]
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
                    self._UKF_raw_measurement_stamp,
                    self._UKF_raw_measurement[QUAD_State.X],
                    self._UKF_raw_measurement[QUAD_State.Y],
                    self._UKF_raw_measurement[QUAD_State.Z],
                    0,
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
                    self._UKF_raw_measurement[QUAD_State.YAW],
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
                    0,

                    # Latency through pipeline
                    self._cam_to_image_lag,
                    self._image_to_transform_lag,
                    self._transform_to_UKF_lag,
                    self._yolo_cam_to_image_lag,
                    self._yolo_image_to_transform_lag,
                    self._yolo_transform_to_UKF_lag,
                    self._UKF_meas_age,
                ]
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
