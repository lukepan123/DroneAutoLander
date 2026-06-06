import csv
from datetime import datetime

import numpy as np
import tf_transformations

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy)
import tf2_ros
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import (State, AttitudeTarget)
from mavros_msgs.srv import (CommandBool, CommandTOL, SetMode, MessageInterval)

from .state_definitions import LP_State, LP_Measurement
from .ukf import UKF
from .mpc_controller import MPCController
from .pid_controller import PIDController


class Orchestrator(Node):
    """ Defines the orchestrator node"""
    def __init__(self):
        super().__init__('orchestrator')
        # ---- STATE VARIABLES ----
        # ---- Global State Variables ----
        self.controller_state = 0

        self._max_runtime = 100.0
        self._boundary_limit = 500.0

        self.state = State()
        self.odometry = Odometry()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.landing_pad_odometry = Odometry()
        self.landing_pad_true_odometry = Odometry()
        self.landing_pad_position = [0,0,0]
        self.landing_pad_velocity = [0,0,0]
        # self._landing_pad_position_lp = LowPassFilter(0.8)
        # self._landing_pad_velocity_lp = LowPassFilter(0.5)
        self.landing_pad_velocity_mag = 0.0
        self.landing_pad_accel_mag = 0.0
        self.landing_pad_yaw = 0.0
        self.landing_pad_yaw_rate = 0.0

        self.target_z = 8.0 # m

        # ---- State 0xxx (Pre-arm) Variables ----
        self._mode_requested = False
        self._mode_confirmed = False
        self._mode = 'GUIDED'
        self._arm_requested = False
        self._armed_confirmed = False
        self._armed_time = None

        # ---- State 1xxx (Take-off) Variables ----
        self._tko_requested = False
        self._tko_reached = False
        self._tko_complete_time = None
        self._tko_altitude_SP = self.target_z # Inititalise at initial target altitude

        # ---- State 2xxx (Searching for Landing Pad) Variables ----
        self._landing_pad_found = False
        self._landing_pad_first_seen_time = None
        self._landing_pad_visual_time_SP = 0.1

        # ---- State 3xxx (Maintaining Landing Pad Lock) Variables ----
        self._landing_pad_locked_time_SP = self._landing_pad_visual_time_SP + 5.0
        self._landing_pad_lost_time = None
        self._landing_pad_lost_time_SP = 2.0

        # ---- State 4xxx (Beginning Landing Descent) Variables ----
        self._landed_time = None
        self.cutoff = False

        # ---- State 5xxx (Landing Abort) Variables ---- 

        # ---- State 6xxx (Landing Confirmed) Variables ---- 
        self._idle_before_RTL_SP = 10.0

        # ---- State 7xxx (RTL) Variables ----
        self._rtl_initiated = False

        # ---- CONTROL CLASS INITIALISATIONS ----
        self._UKF_last_tf_stamp = Time().to_msg()
        self._UKF_last_update = self.get_clock().now()
        self._UKF_timer_rate = 0.01
        self._UKF_filter = UKF(self._UKF_timer_rate)

        self._control_timer_rate = 0.03
        #self._mpc_controller = MPCController()
        self._pid_controller = PIDController(self._control_timer_rate)

        # ---- SUBSCRIPTIONS ----
        _state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._state_sub = self.create_subscription(State, '/mavros/state', self._fcu_state_callback, _state_qos)

        _odom_qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._odometry_sub = self.create_subscription(Odometry, '/mavros/global_position/local', self._odometry_callback, _odom_qos)
        self._twist_sub = self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', self._twist_callback, _odom_qos)
        self._landing_pad_true_odometry_sub = self.create_subscription(Odometry, '/landing_pad/odom', self._landing_pad_true_odometry_callback, _odom_qos)
        self._landing_pad_found_sub = self.create_subscription(Bool, '/landing_pad/found', self._landing_pad_found_callback, 10)

        # ---- PUBLISHERS ----
        self.att_pub = self.create_publisher(AttitudeTarget, '/mavros/setpoint_raw/attitude', 10)

        # ---- TF2 ----
        self._tf_map_landing_pad_buffer = tf2_ros.Buffer()
        self._tf_map_landing_pad_listener = tf2_ros.TransformListener(self._tf_map_landing_pad_buffer, self)
  
        # ---- CLIENTS ----
        self._set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self._arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self._takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self._message_interval_client = self.create_client(MessageInterval, '/mavros/set_message_interval')

        # ---- CONTROL TIMERS ----
        self._UKF_timer = self.create_timer(self._UKF_timer_rate, self._ukf_loop)
        self._safety_timer_rate = 1.0
        self._safety_timer = self.create_timer(self._safety_timer_rate, self._safety_loop)
        self._control_timer = self.create_timer(self._control_timer_rate, self._control_loop)

        #  ---- DIAGNOSTICS AND LOGGING ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_filename = f"controller_{timestamp}.csv"
        self._csv_file = open(self._csv_filename, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'timestamp',
            'quad_x', 'quad_y', 'quad_z', 'quad_vx', 'quad_vy', 'quad_vz',
            'quad_yaw',
            'quad_omgx', 'quad_omgy', 'quad_omgz',
            'landing_pad_x', 'landing_pad_y', 'landing_pad_z', 'landing_pad_vx', 'landing_pad_vy', 'landing_pad_vz', 
            'landing_pad_yaw',
            'landing_pad_true_x', 'landing_pad_true_y', 'landing_pad_true_z', 'landing_pad_true_vx', 'landing_pad_true_vy', 'landing_pad_true_vz',
            'landing_pad_true_yaw'
        ])
        self.get_logger().info(f'CSV logging initialized: {self._csv_filename}')

        self._set_message_intervals()
        self.get_logger().info('Auto Lander (callbacks) started')

    # ---- CALLBACK IMPLEMENTATIONS ----
    def _set_message_intervals(self):
        """ NOT IN USE.
        """


    def _set_single_message_interval(self, message_id, rate, description):
        """ Set a single message interval.
        """
        if not self._message_interval_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'MessageInterval service not ready for {description}')
            return
            
        req = MessageInterval.Request()
        req.message_id = message_id
        req.message_rate = rate
        
        fut = self._message_interval_client.call_async(req)
        fut.add_done_callback(lambda f, desc=description, mid=message_id, r=rate: self._on_message_interval_done(f, desc, mid, r))
        
        self.get_logger().info(f'Setting {description} (ID: {message_id}) to {rate}Hz...')


    def _on_message_interval_done(self, fut, description, message_id, rate):
        """ Handle message interval service response.
        """
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'{description} interval setting exception: {e}')
            return

        if getattr(res, 'success', False):
            self.get_logger().info(f'{description} interval set to {rate}Hz successfully')
        else:
            self.get_logger().warn(f'{description} interval setting failed (ID: {message_id}, Rate: {rate}Hz)')


    def _fcu_state_callback(self, msg: State):
        """ Confirm state from FCU, and if good arm the FCU.
        """
        self.state = msg

        if self.state.mode == self._mode and not self._mode_confirmed:
            self._mode_confirmed = True
            self.get_logger().info(f'{self._mode} confirmed by FCU.')

        if self.state.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = self.get_clock().now().nanoseconds / 1e9


    def _odometry_callback(self, msg: Odometry):
        """ Obtain FCU Odometry, process quaternions into euler roll, pitch and yaw values. 
            Also monitor takeoff condition.
        """
        self.odometry = msg
        alt = msg.pose.pose.position.z

        q = self.odometry.pose.pose.orientation
        self.roll, self.pitch, self.yaw = tf_transformations.euler_from_quaternion([
            q.x, q.y, q.z, q.w
        ])
        
        if self._armed_confirmed and not self._tko_reached and alt > (self._tko_altitude_SP - 0.5):
            self._tko_reached = True
            self._tko_complete_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f'Takeoff complete at {alt:.2f} m - Starting {self._max_runtime}s safety timer')


    def _twist_callback(self, msg: TwistStamped):
        """ Obtain FCU angular rates.
        """
        self.odometry.twist.twist.angular.x = msg.twist.angular.x
        self.odometry.twist.twist.angular.y = msg.twist.angular.y
        self.odometry.twist.twist.angular.z = msg.twist.angular.z


    def _landing_pad_found_callback(self, msg: Bool):
        """ Obtain landing pad found signal from landing_pad_detector node.
        """
        self._landing_pad_found = msg.data


    def _landing_pad_true_odometry_callback(self, msg: Odometry):
        """ SITL ONLY: Pass true odometry of landing pad for data.
        """
        self.landing_pad_true_odometry = msg


    def _request_mode(self):
        """ Set mode on FCU.
        """
        if not self._set_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('SetMode service not ready yet.')
            return
        self._mode_requested = True
        req = SetMode.Request()
        req.custom_mode = self._mode
        fut = self._set_mode_client.call_async(req)
        fut.add_done_callback(self._on_set_mode_done)
        self.get_logger().info(f'Requesting {self._mode}...')


    def _on_set_mode_done(self, fut):
        """ Check if mode properly set on FCU.
        """
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'SetMode exception: {e}')
            self._mode_requested = False
            return

        if getattr(res, 'mode_sent', False):
            self.get_logger().info(f'{self._mode} command accepted (awaiting FCU report).')
        else:
            self.get_logger().error(f'{self._mode} command rejected by FCU.')
            self._mode_requested = False


    def _request_arm(self):
        """ Request FCU Arm.
        """
        if not self._arming_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Arming service not ready yet.')
            return
        self._arm_requested = True
        req = CommandBool.Request()
        req.value = True
        fut = self._arming_client.call_async(req)
        fut.add_done_callback(self._on_arm_done)
        self.get_logger().info('Requesting ARM...')


    def _on_arm_done(self, fut):
        """ Check if FCU Armed.
        """
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'Arming exception: {e}')
            self._arm_requested = False
            return

        if getattr(res, 'success', False):
            self.get_logger().info('Arm accepted (awaiting FCU armed=true).')
        else:
            self.get_logger().error(f'Arm rejected by FCU (result={getattr(res, "result", None)}).')
            self._arm_requested = False


    def _request_takeoff(self):
        """ Request FCU takeoff.
        """
        if not self._takeoff_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Takeoff service not ready yet.')
            return
        self._tko_requested = True
        req = CommandTOL.Request()
        req.altitude = float(self._tko_altitude_SP)
        fut = self._takeoff_client.call_async(req)
        fut.add_done_callback(self._on_takeoff_done)
        self.get_logger().info(f'Requesting takeoff to {self._tko_altitude_SP:.1f} m...')


    def _on_takeoff_done(self, fut):
        """ Confirm FCU takeoff successful.
        """
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'Takeoff exception: {e}')
            self._tko_requested = False
            return

        if getattr(res, 'success', False):
            self.get_logger().info('Takeoff command accepted (monitoring altitude).')
        else:
            self.get_logger().error('Takeoff rejected by FCU.')
            self._tko_requested = False

    def _initiate_rtl(self, reason):
        """ Initiate return to land mode and disable tracking.
        """
        if self._rtl_initiated:
            return
            
        self._rtl_initiated = True
        self._tracking_enabled = False
        
        self.get_logger().warn(f'SAFETY: Initiating RTL due to {reason}')
        
        if not self._set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('SetMode service not available for RTL!')
            return
            
        req = SetMode.Request()
        req.custom_mode = 'RTL'
        fut = self._set_mode_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_rtl_done(f, reason))


    def _on_rtl_done(self, fut, reason):
        """ Handle RTL mode change response.
        """
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'RTL exception: {e}')
            return

        if getattr(res, 'mode_sent', False):
            self.get_logger().info(f'RTL command accepted (reason: {reason})')
        else:
            self.get_logger().error(f'RTL command rejected by FCU (reason: {reason})')


    def _ukf_loop(self):
        """ Run UKF predict and update loops.
        """
        now = self.get_clock().now()
        dt = (now - self._UKF_last_update).nanoseconds * 1e-9

        # Predict UKF step
        self._UKF_filter.predict(dt)

        # Update UKF step
        try:
            tf_msg = self._tf_map_landing_pad_buffer.lookup_transform('map', 'landing_pad_link', Time())

            stamp_is_new = (
                tf_msg.header.stamp.sec   != self._UKF_last_tf_stamp.sec or
                tf_msg.header.stamp.nanosec != self._UKF_last_tf_stamp.nanosec
            )

            if stamp_is_new:
                # Extract pose measurement
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                roll, pitch, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
                state = np.array([t.x, t.y, t.z, yaw])

                # Update measurement noise based on drone angular rates
                cov_adj_x = 2 * (1 + np.sqrt(self.odometry.twist.twist.angular.y**2 + self.odometry.twist.twist.angular.z**2))
                cov_adj_y = 2 * (1 + np.sqrt(self.odometry.twist.twist.angular.x**2 + self.odometry.twist.twist.angular.z**2))
                cov_adj_z = 2 * (1 + np.sqrt(self.odometry.twist.twist.angular.x**2 + self.odometry.twist.twist.angular.y**2))

                self._UKF_filter.R = np.diag([
                    0.01 * cov_adj_x, 0.01 * cov_adj_y, 0.01 * cov_adj_z,
                    0.01 * cov_adj_z
                ])

                accepted = self._UKF_filter.update(state)
            
            self._UKF_last_tf_stamp = tf_msg.header.stamp

        except Exception:
            pass  # predict-only cycle, no correction this tick

        # Always publish current UKF state
        x = self._UKF_filter.x

        self.landing_pad_position = np.array([x[LP_State.PX], x[LP_State.PY], x[LP_State.PZ]])

        yaw = x[LP_State.YAW]
        vx = x[LP_State.V] * np.cos(yaw)
        vy = x[LP_State.V] * np.sin(yaw)
        self.landing_pad_velocity = np.array([vx, vy, 0.0])

        self.landing_pad_odometry.header.stamp = self.get_clock().now().to_msg()
        self.landing_pad_odometry.header.frame_id = 'map'

        self.landing_pad_odometry.pose.pose.position.x = x[LP_State.PX]
        self.landing_pad_odometry.pose.pose.position.y = x[LP_State.PY]
        self.landing_pad_odometry.pose.pose.position.z = x[LP_State.PZ]

        quat = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
        self.landing_pad_odometry.pose.pose.orientation.x = quat[0]
        self.landing_pad_odometry.pose.pose.orientation.y = quat[1]
        self.landing_pad_odometry.pose.pose.orientation.z = quat[2]
        self.landing_pad_odometry.pose.pose.orientation.w = quat[3]

        self.landing_pad_yaw = yaw

        self.landing_pad_odometry.twist.twist.linear.x = vx
        self.landing_pad_odometry.twist.twist.linear.y = vy
        self.landing_pad_odometry.twist.twist.linear.z = 0.0

        # Update dt
        self._UKF_last_update = now

        # Check for timer overruns
        end = self.get_clock().now()
        elapsed = (end - now).nanoseconds / 1e6
        if elapsed > self._UKF_timer_rate * 1000:
            self.get_logger().warn(f"UKF loop took {elapsed:.2f} ms!")

    def _control_loop(self):
        start = self.get_clock().now()

        # ---- State 0000 (Pre-arm) - Below conditions need to be met ALWAYS so we check regardless of state
        if not self.state.connected:
            self.controller_state = 0
            return

        if not self._mode_confirmed:
            if not self._mode_confirmed:
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
            self.controller_state = 7000
            return
        
        if self._armed_confirmed and self._armed_time is not None and not self._tko_requested:
            self.controller_state = 1000

        # ---- State 1000 (Start Takeoff)
        if self.controller_state == 1000:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._armed_time
            if elapsed < 5.0:
                if not hasattr(self, '_armed_wait_logged') or not self._armed_wait_logged:
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
                self._landing_pad_first_seen_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info(f"Landing Pad found, starting {self._landing_pad_visual_time_SP} sec timer...")
                self.controller_state = 2100

        # ---- State 2100 (Maintain Landing Pad Visual Lock)
        if self.controller_state == 2100:
            now = self.get_clock().now().nanoseconds / 1e9
            if (now - self._landing_pad_first_seen_time) > self._landing_pad_visual_time_SP:
                self.get_logger().info(f"Landing Pad visual hold ok, starting {self._landing_pad_locked_time_SP} sec timer...")
                self.controller_state = 3000
        
        # ---- State 3000 (Move over and Maintain Landing Pad Lock)
        if self.controller_state == 3000:
            now = self.get_clock().now().nanoseconds / 1e9
            if self._landing_pad_found:
                self._landing_pad_lost_time = None
                if (now - self._landing_pad_first_seen_time) > self._landing_pad_locked_time_SP:
                    self.get_logger().info("Landing Pad Acquired")
                    self.controller_state = 4000
            else:
                if self._landing_pad_lost_time is None:
                    self._landing_pad_lost_time = now
                else:
                    if (now - self._landing_pad_lost_time) > self._landing_pad_lost_time_SP:
                        self._landing_pad_first_seen_time = None
                        self._landing_pad_lost_time = None
                        self.get_logger().info("Landing Pad Lost!")
                        self.controller_state = 2000

        # ---- State 4000 (Begin Landing Descent)
        if self.controller_state == 4000:
            now = self.get_clock().now().nanoseconds / 1e9
            self.target_z = 0.2 #land

            # Check target error is not bad
            err_x = abs(self.landing_pad_position[0] - self.odometry.pose.pose.position.x)
            err_y = abs(self.landing_pad_position[1] - self.odometry.pose.pose.position.y)

            if self.odometry.pose.pose.position.z <= 0.8 and err_x < 0.2 and err_y < 0.2:
                self.cutoff = True
                self._landed_time = now
                self.get_logger().info("Throttle Cut Engaged")
                self.controller_state = 6000
            elif self.odometry.pose.pose.position.z <= 0.6 and (err_x >= 0.2 or err_y >= 0.2):
                self.target_z = 5.0
                self.get_logger().info("Landing Aborted - Trying Again")
                self.controller_state = 3000

        # ---- State 5000 (Landing Failed - Try again)

        # ---- State 6000 (Landing Success - Idle Until RTL)
        if self.controller_state == 6000:
            now = self.get_clock().now().nanoseconds / 1e9
            if (now - self._landed_time) > self._idle_before_RTL_SP:
                self._landing_pad_first_seen_time = None
                self._landing_pad_lost_time = None
                self.get_logger().info("Landing Pad Lost!")
                self.controller_state = 6100

        # ---- State 6100 (Initiate RTL)
        if self.controller_state == 6100:
            self._initiate_rtl("Landing Complete")

        # ---- Run Controller ----
        if self.controller_state >= 3000 and self.controller_state <= 6000:
           #msg = self._mpc_controller.compute_control(self)
           msg = self._pid_controller.update(self)


        # ---- Data logging ----
        lp_pad_true_roll, lp_pad_true_pitch, lp_pad_true_yaw = tf_transformations.euler_from_quaternion([
            self.landing_pad_true_odometry.pose.pose.orientation.x, 
            self.landing_pad_true_odometry.pose.pose.orientation.y, 
            self.landing_pad_true_odometry.pose.pose.orientation.z, 
            self.landing_pad_true_odometry.pose.pose.orientation.w
        ])

        current_time = self.get_clock().now().nanoseconds / 1e9
        self._csv_writer.writerow([
            current_time,
            self.odometry.pose.pose.position.x,
            self.odometry.pose.pose.position.y, 
            self.odometry.pose.pose.position.z,
            self.odometry.twist.twist.linear.x,
            self.odometry.twist.twist.linear.y,
            self.odometry.twist.twist.linear.z,
            self.yaw,
            self.odometry.twist.twist.angular.x,
            self.odometry.twist.twist.angular.y,
            self.odometry.twist.twist.angular.z,
            self.landing_pad_odometry.pose.pose.position.x,
            self.landing_pad_odometry.pose.pose.position.y,
            self.landing_pad_odometry.pose.pose.position.z,
            self.landing_pad_odometry.twist.twist.linear.x,
            self.landing_pad_odometry.twist.twist.linear.y,
            self.landing_pad_odometry.twist.twist.linear.z,
            self.landing_pad_yaw,
            self.landing_pad_true_odometry.pose.pose.position.x,
            self.landing_pad_true_odometry.pose.pose.position.y,
            self.landing_pad_true_odometry.pose.pose.position.z,
            # Fix velocity because it is given in body frame not world frame...""
            self.landing_pad_true_odometry.twist.twist.linear.x * np.cos(lp_pad_true_yaw),
            self.landing_pad_true_odometry.twist.twist.linear.x * np.sin(lp_pad_true_yaw),
            self.landing_pad_true_odometry.twist.twist.linear.z,
            lp_pad_true_yaw,
        ])
        self._csv_file.flush()

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
            if elapsed_since_takeoff >= self._max_runtime:
                self.get_logger().warn(f'{self._max_runtime} seconds elapsed since takeoff - Initiating RTL')
                self._initiate_rtl(f'{self._max_runtime}-second timer expired')
                return
        
        # Check boundary conditions
        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y
        
        if abs(x) > self._boundary_limit or abs(y) > self._boundary_limit:
            self.get_logger().warn(f'Boundary violation: position ({x:.1f}, {y:.1f}) - Initiating RTL')
            self._initiate_rtl(f'boundary violation at ({x:.1f}, {y:.1f})')
            return

    def destroy_node(self):
        """Clean up CSV file when node is destroyed"""
        if hasattr(self, 'csv_file'):
            self._csv_file.close()
            self.get_logger().info(f'CSV file closed: {self._csv_filename}')
        super().destroy_node()

# ---- HELPER CLASSES ----
class LowPassFilter:
    def __init__(self, cutoff_hz):
        self.cutoff = cutoff_hz
        self.prev_values = None
        self.prev_stamp = None

    def update(self, values, stamp):
        # Initialise if first time
        if self.prev_values is None:
            self.prev_values = values.copy()
            self.prev_stamp = stamp
            return values

        dt = (stamp - self.prev_stamp).nanoseconds * 1e-9
        if dt <= 0.0:
            return self.prev_values

        tau = 1.0 / (2.0 * np.pi * self.cutoff)
        alpha = dt / (tau + dt)

        filtered_values = self.prev_values + alpha * (values - self.prev_values)

        self.prev_values = filtered_values.copy()
        self.prev_stamp = stamp
        return filtered_values


# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
