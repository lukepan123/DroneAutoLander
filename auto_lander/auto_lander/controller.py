#!/usr/bin/env python3
import math
import csv
import numpy as np
from datetime import datetime

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
import tf2_ros

from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, MessageInterval

from .implimented_controllers import Controller, MPCController

class ChaserController(Node):
    def __init__(self):
        super().__init__('chaser_controller')

        # ---------------- PUBLISHERS AND SUBSCRIPTIONS ----------------
        # Initialise MAVROS subscriptions
        state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.state_sub = self.create_subscription(State, '/mavros/state', self._on_state, state_qos)

        pose_qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._on_pose, pose_qos)
        self.twist_sub = self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', self._on_twist, pose_qos)        
        self.global_pos_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self._on_global_position, pose_qos)
        self.compass_sub = self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self._on_compass, pose_qos)

        # Initialise target frame listener
        self.tf_map_target_buffer = tf2_ros.Buffer()
        self.tf_map_target_listener = tf2_ros.TransformListener(self.tf_map_target_buffer, self)
        self.timer = self.create_timer(0.1, self._tf_map_to_target_callback)

        # Initialise publisher and subscription to target odometry UKF
        self.target_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/target_pose', 10)
        self.target_odometry_sub = self.create_subscription(Odometry, '/odometry/filtered', self._on_target_odometry, 10)
        self.tracking_enable_sub = self.create_subscription(Bool, '/tracking_enable', self._on_tracking_enable, 10)
        self.target_found_sub = self.create_subscription(Bool, '/target_found', self._on_target_found, 10)

        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        
        # LP Filters on target pose/vel
        self.target_pose = None
        self.target_vel = None
        self.target_pose_lp = LowPassFilter(0.8)
        self.target_vel_lp = LowPassFilter(0.5)
        self.target_found = False

        # ---------------- QUADCOPTER STATE ----------------
        # Initialise node variables
        self.state = State()
        self.pose = PoseStamped()
        self.twist = TwistStamped()
        self.target_odometry = Odometry()
        self.global_pos = NavSatFix()
        self.compass_hdg = 0.0
        
        # Initialise controller variables
        self.target_altitude = 15.0
        self.kp = 0.7
        self.ki = 0.0
        self.kd = 0.0
        self.x_error_int = 0.0
        self.y_error_int = 0.0
        self.prev_x_meas = 0.0
        self.prev_y_meas = 0.0
        self.prev_time = self.get_clock().now()

        self.x_p = 0.0
        self.x_i = 0.0
        self.x_d = 0.0
        self.y_p = 0.0
        self.y_i = 0.0
        self.y_d = 0.0

        # Initialize MPC
        self.mpc_controller = MPCController(self)

        # Initialise MAVROS publishers and control loop
        
        self.setpoint_timer = self.create_timer(0.10, self._control_loop) # <- control loop (10Hz)

        # Initialise MAVROS clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.message_interval_client = self.create_client(MessageInterval, '/mavros/set_message_interval')

        # Initialise flight .csv log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"controller_{timestamp}.csv"
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'quad_x', 'quad_y', 'quad_z', 'quad_vx', 'quad_vy', 'quad_vz', 'quad_omgx', 'quad_omgy', 'quad_omgz', 
            'target_x', 'target_y', 'target_z', 'target_vx', 'target_vy', 'target_vz',
            'target_lpx', 'target_lpy', 'target_lpz', 'target_lpvx', 'target_lpvy', 'target_lpvz'
            # 'x_p', 'x_i', 'x_d', 'y_p', 'y_i', 'y_d'
        ])
        self.get_logger().info(f'CSV logging initialized: {self.csv_filename}')

        # Initialise arming and landing variables
        self._guided_requested = False
        self._guided_confirmed = False

        self._arm_requested = False
        self._armed_confirmed = False

        self._tko_requested = False
        self._tko_reached = False

        self._tracking_enabled = False
        self._tracking_manual_override = False
        self._armed_time = None
        self._takeoff_complete_time = None
        self._rtl_initiated = False

        self.boundary_limit = 25.0  # 15m from center in any direction
        
        self.orchestrator = self.create_timer(0.2, self._orchestrate)
        self.safety_timer = self.create_timer(1.0, self._check_safety_conditions)

        # Set MAVROS message intervals
        self._set_message_intervals()

        self.get_logger().info('Auto Lander (callbacks) started')

    def _set_message_intervals(self):
        """Set MAVROS message intervals for global position and compass to 100.0Hz"""
        self._set_single_message_interval(32, 100.0, "Local Position") 
        self._set_single_message_interval(33, 100.0, "Global Position") 
        self._set_single_message_interval(74, 100.0, "Compass Heading")

    def _set_single_message_interval(self, message_id, rate, description):
        """Set a single message interval"""
        if not self.message_interval_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'MessageInterval service not ready for {description}')
            return
            
        req = MessageInterval.Request()
        req.message_id = message_id
        req.message_rate = rate
        
        fut = self.message_interval_client.call_async(req)
        fut.add_done_callback(lambda f, desc=description, mid=message_id, r=rate: self._on_message_interval_done(f, desc, mid, r))
        
        self.get_logger().info(f'Setting {description} (ID: {message_id}) to {rate}Hz...')

    def _on_message_interval_done(self, fut, description, message_id, rate):
        """Handle message interval service response"""
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'{description} interval setting exception: {e}')
            return

        if getattr(res, 'success', False):
            self.get_logger().info(f'{description} interval set to {rate}Hz successfully')
        else:
            self.get_logger().warn(f'{description} interval setting failed (ID: {message_id}, Rate: {rate}Hz)')

    def _on_state(self, msg: State):
        self.state = msg

        if self.state.mode == 'GUIDED' and not self._guided_confirmed:
            self._guided_confirmed = True
            self.get_logger().info('GUIDED confirmed by FCU.')

        if self.state.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = self.get_clock().now().nanoseconds / 1e9
            
            compass_rad = math.radians(self.compass_hdg)
            
            target_x = self.pose.pose.position.x + 7.0 * math.sin(compass_rad)
            target_y = self.pose.pose.position.y + 7.0 * math.cos(compass_rad)
            
            self.estimated_state = np.array([target_x, target_y])
            
            self.get_logger().info(f'Armed confirmed by FCU. Compass: {self.compass_hdg:.1f}°')
            self.get_logger().info(f'Estimated target at: ({target_x:.2f}, {target_y:.2f})')


    # Helper functions for subscriptions
    def _on_pose(self, msg: PoseStamped):
        self.pose = msg
        alt = msg.pose.position.z
        
        if self._armed_confirmed and not self._tko_reached and alt > (self.target_altitude - 0.5):
            self._tko_reached = True
            self._takeoff_complete_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f'Takeoff complete at {alt:.2f} m - Starting 60s safety timer')
            self._enable_tracking()

    def _on_twist(self, msg: TwistStamped):
        self.twist = msg

    def _on_global_position(self, msg: NavSatFix):
        self.global_pos = msg

    def _on_compass(self, msg: Float64):
        self.compass_hdg = float(msg.data)

    def _on_target_odometry(self, msg: Odometry):
        self.target_odometry = msg

    def _on_tracking_enable(self, msg: Bool):
        """Handle manual tracking enable/disable commands"""
        self._tracking_manual_override = msg.data
        if msg.data:
            self._enable_tracking_manual()
        else:
            self._disable_tracking_manual()
        self.get_logger().info(f'Manual tracking override: {"ENABLED" if msg.data else "DISABLED"}')

    def _on_target_found(self, msg: Bool):
        self.target_found = msg.data
    # -------------------------------------------------------------


    def _orchestrate(self):
        if not self.state.connected:
            return

        if not self._guided_confirmed:
            if not self._guided_requested:
                self._request_guided()
            return

        if not self._armed_confirmed:
            if not self._arm_requested:
                self._request_arm()
            return

        if self._armed_confirmed and self._armed_time is not None and not self._tko_requested:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._armed_time
            if elapsed < 5.0:
                if not hasattr(self, '_armed_wait_logged') or not self._armed_wait_logged:
                    self.get_logger().info("Armed. Waiting 5s before takeoff...")
                    self._armed_wait_logged = True
                return
            else:
                self._request_takeoff()
                self._armed_wait_logged = False
                return

    def _request_guided(self):
        if not self.set_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('SetMode service not ready yet.')
            return
        self._guided_requested = True
        req = SetMode.Request()
        req.custom_mode = 'GUIDED'
        fut = self.set_mode_client.call_async(req)
        fut.add_done_callback(self._on_set_mode_done)
        self.get_logger().info('Requesting GUIDED...')

    def _on_set_mode_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'SetMode exception: {e}')
            self._guided_requested = False
            return

        if getattr(res, 'mode_sent', False):
            self.get_logger().info('GUIDED command accepted (awaiting FCU report).')
        else:
            self.get_logger().error('GUIDED command rejected by FCU.')
            self._guided_requested = False

    def _request_arm(self):
        if not self.arming_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Arming service not ready yet.')
            return
        self._arm_requested = True
        req = CommandBool.Request()
        req.value = True
        fut = self.arming_client.call_async(req)
        fut.add_done_callback(self._on_arm_done)
        self.get_logger().info('Requesting ARM...')

    def _on_arm_done(self, fut):
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
        if not self.takeoff_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Takeoff service not ready yet.')
            return
        self._tko_requested = True
        req = CommandTOL.Request()
        req.altitude = float(self.target_altitude)
        fut = self.takeoff_client.call_async(req)
        fut.add_done_callback(self._on_takeoff_done)
        self.get_logger().info(f'Requesting takeoff to {self.target_altitude:.1f} m...')

    def _on_takeoff_done(self, fut):
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

    def _enable_tracking(self):
        if not self._tracking_enabled:
            self._tracking_enabled = True
            self.get_logger().info('Tracking enabled (automatic - takeoff complete).')

    def _enable_tracking_manual(self):
        """Enable tracking manually via topic command"""
        if not self._tracking_enabled:
            self._tracking_enabled = True
            self.get_logger().info('Tracking enabled (manual override).')

    def _disable_tracking_manual(self):
        """Disable tracking manually via topic command"""
        if self._tracking_enabled:
            self._tracking_enabled = False
            self.get_logger().info('Tracking disabled (manual override).')

    def _check_safety_conditions(self):
        """Check safety conditions and initiate RTL if necessary"""
        if self._rtl_initiated or not self._tko_reached:
            return
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check 120-second timer after takeoff
        if self._takeoff_complete_time is not None:
            elapsed_since_takeoff = current_time - self._takeoff_complete_time
            if elapsed_since_takeoff >= 60.0:
                self.get_logger().warn('120 seconds elapsed since takeoff - Initiating RTL')
                self._initiate_rtl('120-second timer expired')
                return
        
        # Check boundary conditions (30x30m square)
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        if abs(x) > self.boundary_limit or abs(y) > self.boundary_limit:
            self.get_logger().warn(f'Boundary violation: position ({x:.1f}, {y:.1f}) - Initiating RTL')
            self._initiate_rtl(f'boundary violation at ({x:.1f}, {y:.1f})')
            return

    def _initiate_rtl(self, reason):
        """Initiate return to land mode and disable tracking"""
        if self._rtl_initiated:
            return
            
        self._rtl_initiated = True
        self._tracking_enabled = False
        
        self.get_logger().warn(f'SAFETY: Initiating RTL due to {reason}')
        
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('SetMode service not available for RTL!')
            return
            
        req = SetMode.Request()
        req.custom_mode = 'RTL'
        fut = self.set_mode_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_rtl_done(f, reason))

    def _on_rtl_done(self, fut, reason):
        """Handle RTL mode change response"""
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'RTL exception: {e}')
            return

        if getattr(res, 'mode_sent', False):
            self.get_logger().info(f'RTL command accepted (reason: {reason})')
        else:
            self.get_logger().error(f'RTL command rejected by FCU (reason: {reason})')

    def _tf_map_to_target_callback(self):
        try:
            map_to_target_transform = self.tf_map_target_buffer.lookup_transform(
            'map',
            'target_link',
            rclpy.time.Time()   # latest available transform
        )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        
        pose_msg = PoseWithCovarianceStamped()

        # Header
        pose_msg.header.stamp = map_to_target_transform.header.stamp
        pose_msg.header.frame_id = map_to_target_transform.header.frame_id

        # Position
        pose_msg.pose.pose.position.x = map_to_target_transform.transform.translation.x
        pose_msg.pose.pose.position.y = map_to_target_transform.transform.translation.y
        pose_msg.pose.pose.position.z = map_to_target_transform.transform.translation.z

        # Orientation
        pose_msg.pose.pose.orientation = map_to_target_transform.transform.rotation

        # Covariance (you MUST set this for UKF stability)

        # Increase this when angular velcoity is high!
        cov_multiplier_x = 1 + 5 * np.sqrt(self.twist.twist.angular.y**2 + self.twist.twist.angular.z**2)
        cov_multiplier_y = 1 + 5 * np.sqrt(self.twist.twist.angular.x**2 + self.twist.twist.angular.z**2)
        cov_multiplier_z = 1 + 5 * np.sqrt(self.twist.twist.angular.x**2 + self.twist.twist.angular.y**2)

        cov = np.zeros((6, 6))
        cov[0, 0] = 0.15 * cov_multiplier_x   # x
        cov[1, 1] = 0.15 * cov_multiplier_y   # y
        cov[2, 2] = 0.15 * cov_multiplier_x   # z
        cov[3, 3] = 0.4   # roll
        cov[4, 4] = 0.4   # pitch
        cov[5, 5] = 0.2   # yaw

        pose_msg.pose.covariance = cov.flatten().tolist()

        self.target_pose_pub.publish(pose_msg)

    def _control_loop(self):
        # Don't write to CSV nor send commands if rtl initiated
        if self._rtl_initiated:
            return

        # Update LP Filters
        stamp = self.get_clock().now()

        target_pose_raw = np.array([
            self.target_odometry.pose.pose.position.x, 
            self.target_odometry.pose.pose.position.y, 
            self.target_odometry.pose.pose.position.z])
        self.target_pose = self.target_pose_lp.update(target_pose_raw, stamp)

        target_vel_raw = np.array([
            self.target_odometry.twist.twist.linear.x, 
            self.target_odometry.twist.twist.linear.y, 
            self.target_odometry.twist.twist.linear.z])
        self.target_vel = self.target_vel_lp.update(target_vel_raw, stamp)
        
        # Run control loop
        # msg = Controller(self)

        # Inside your control loop / timer callback
        msg = self.mpc_controller.compute_control(self)

        current_time = self.get_clock().now().nanoseconds / 1e9
        self.csv_writer.writerow([
            current_time,
            self.pose.pose.position.x,
            self.pose.pose.position.y, 
            self.pose.pose.position.z,
            self.twist.twist.linear.x,
            self.twist.twist.linear.y,
            self.twist.twist.linear.z,
            self.twist.twist.angular.x,
            self.twist.twist.angular.y,
            self.twist.twist.angular.z,
            self.target_odometry.pose.pose.position.x,
            self.target_odometry.pose.pose.position.y,
            self.target_odometry.pose.pose.position.z,
            self.target_odometry.twist.twist.linear.x,
            self.target_odometry.twist.twist.linear.y,
            self.target_odometry.twist.twist.linear.z,
            self.target_pose[0],
            self.target_pose[1],
            self.target_pose[2],
            self.target_vel[0],
            self.target_vel[1],
            self.target_vel[2],
            # self.x_p,
            # self.x_i,
            # self.x_d,
            # self.y_p,
            # self.y_i,
            # self.y_d
        ])
        self.csv_file.flush()


    def destroy_node(self):
        """Clean up CSV file when node is destroyed"""
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            self.get_logger().info(f'CSV file closed: {self.csv_filename}')
        super().destroy_node()

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

# ---------------- MAIN ----------------
def main(args=None):
    rclpy.init(args=args)
    node = ChaserController()
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
