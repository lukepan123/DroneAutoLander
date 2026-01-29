#!/usr/bin/env python3
import csv
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
import tf_transformations

from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, MessageInterval

from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

from .implimented_controllers import MPCController

class ChaserController(Node):
    def __init__(self):
        super().__init__('chaser_controller')

        # ---------------- PUBLISHERS AND SUBSCRIPTIONS ----------------
        # Initialise MAVROS subscriptions
        state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.state_sub = self.create_subscription(State, '/mavros/state', self._on_state, state_qos)

        odom_qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.odometry_sub = self.create_subscription(Odometry, '/mavros/global_position/local', self._on_odometry, odom_qos)      

        # Initialise target localisation
        self.tf_map_target_buffer = tf2_ros.Buffer()
        self.tf_map_target_listener = tf2_ros.TransformListener(self.tf_map_target_buffer, self)
        self.filter = UKF(0.04)
        self.UKF_timer = self.create_timer(0.04, self._tf_map_to_target_callback)

        # Initialise publisher and subscription to target odometry UKF
        self.tracking_enable_sub = self.create_subscription(Bool, '/tracking_enable', self._on_tracking_enable, 10)
        self.target_found_sub = self.create_subscription(Bool, '/target_found', self._on_target_found, 10)

        # Publisher for quadcopter velcoity commands
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        
        # ---------------- QUADCOPTER STATE ----------------
        # LP Filters on target position/velocity
        self.target_position = None
        self.target_velocity = None
        self.target_position_lp = LowPassFilter(0.8)
        self.target_velocity_lp = LowPassFilter(0.5)

        self.target_velocity_mag = 0.0
        self.target_yaw = 0.0
        self.target_yaw_rate = 0.0
        self.target_found = False

        # Initialise node variables
        self.state = State()
        self.odometry = Odometry()
        self.target_odometry = Odometry()
        self.yaw = 0.0
        
        # Initialise controller variables
        self.target_altitude = 15.0
        self.prev_time = self.get_clock().now()

        # Initialize MPC and control loop
        self.control_rate = 0.15
        self.mpc_controller = MPCController(self.control_rate)
        self.setpoint_timer = self.create_timer(self.control_rate, self._control_loop) # <- control loop (6.66Hz)

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
            'target_x', 'target_y', 'target_z', 'target_vx', 'target_vy', 'target_vz', 'target_yaw',
            'target_lpx', 'target_lpy', 'target_lpz', 'target_lpvx', 'target_lpvy', 'target_lpvz'
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

    # Helper functions for subscriptions
    def _on_state(self, msg: State):
        self.state = msg

        if self.state.mode == 'GUIDED' and not self._guided_confirmed:
            self._guided_confirmed = True
            self.get_logger().info('GUIDED confirmed by FCU.')

        if self.state.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = self.get_clock().now().nanoseconds / 1e9

    def _on_odometry(self, msg: Odometry):
        self.odometry = msg
        alt = msg.pose.pose.position.z
        
        if self._armed_confirmed and not self._tko_reached and alt > (self.target_altitude - 0.5):
            self._tko_reached = True
            self._takeoff_complete_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f'Takeoff complete at {alt:.2f} m - Starting 60s safety timer')
            self._enable_tracking()

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
        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y
        
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
            tf_msg = self.tf_map_target_buffer.lookup_transform(
                'map',
                'target_link',
                rclpy.time.Time()
            )
        except Exception:
            return

        # Extract pose
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        roll, pitch, yaw = tf_transformations.euler_from_quaternion([
            q.x, q.y, q.z, q.w
        ])

        z = np.array([
            t.x,
            t.y,
            t.z,
            yaw
        ])

        # Predict → Update (correct order)
        self.filter.ukf.predict()
        self.filter.ukf.update(z)

        x = self.filter.ukf.x

        # Publish filtered pose
        self.target_odometry.header.stamp = tf_msg.header.stamp
        self.target_odometry.header.frame_id = 'map'

        self.target_odometry.pose.pose.position.x = x[0]
        self.target_odometry.pose.pose.position.y = x[1]
        self.target_odometry.pose.pose.position.z = x[2]

        quat = tf_transformations.quaternion_from_euler(0.0, 0.0, x[4])
        self.target_odometry.pose.pose.orientation.x = quat[0]
        self.target_odometry.pose.pose.orientation.y = quat[1]
        self.target_odometry.pose.pose.orientation.z = quat[2]
        self.target_odometry.pose.pose.orientation.w = quat[3]

        vx = x[3] * np.cos(x[4])
        vy = x[3] * np.sin(x[4])

        self.target_velocity_mag = x[3]
        self.target_yaw = x[4]
        self.target_yaw_rate = x[5]

        self.target_odometry.twist.twist.linear.x = vx
        self.target_odometry.twist.twist.linear.y = vy
        self.target_odometry.twist.twist.linear.z = 0.0


    def _control_loop(self):
        # Don't write to CSV nor send commands if rtl initiated
        if self._rtl_initiated:
            return

        # Update LP Filters
        stamp = self.get_clock().now()

        target_position_raw = np.array([
            self.target_odometry.pose.pose.position.x, 
            self.target_odometry.pose.pose.position.y, 
            self.target_odometry.pose.pose.position.z])
        self.target_position = self.target_position_lp.update(target_position_raw, stamp)

        target_velocity_raw = np.array([
            self.target_odometry.twist.twist.linear.x, 
            self.target_odometry.twist.twist.linear.y, 
            self.target_odometry.twist.twist.linear.z])
        self.target_velocity = self.target_velocity_lp.update(target_velocity_raw, stamp)

        q = self.odometry.pose.pose.orientation
        _, _, self.yaw = tf_transformations.euler_from_quaternion([
            q.x, q.y, q.z, q.w
        ])
        
        # Run control loop
        msg = self.mpc_controller.compute_control(self)

        current_time = self.get_clock().now().nanoseconds / 1e9
        self.csv_writer.writerow([
            current_time,
            self.odometry.pose.pose.position.x,
            self.odometry.pose.pose.position.y, 
            self.odometry.pose.pose.position.z,
            self.odometry.twist.twist.linear.x,
            self.odometry.twist.twist.linear.y,
            self.odometry.twist.twist.linear.z,
            self.odometry.twist.twist.angular.x,
            self.odometry.twist.twist.angular.y,
            self.odometry.twist.twist.angular.z,
            self.target_odometry.pose.pose.position.x,
            self.target_odometry.pose.pose.position.y,
            self.target_odometry.pose.pose.position.z,
            self.target_odometry.twist.twist.linear.x,
            self.target_odometry.twist.twist.linear.y,
            self.target_odometry.twist.twist.linear.z,
            self.target_yaw,
            self.target_position[0],
            self.target_position[1],
            self.target_position[2],
            self.target_velocity[0],
            self.target_velocity[1],
            self.target_velocity[2],
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
    

class UKF:
    def __init__(self, dt):
        self.dt = dt
        self.dim_x = 6
        self.dim_z = 4

        points = MerweScaledSigmaPoints(
            n=self.dim_x,
            alpha=0.3,
            beta=2.0,
            kappa=0.0
        )

        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            fx=self.fx,
            hx=self.hx,
            dt=dt,
            points=points
        )

        # State: [px, py, pz, v, yaw, yaw_rate]
        self.ukf.x = np.zeros(self.dim_x)

        # Covariance
        self.ukf.P = np.diag([
            0.2, 0.2, 0.2,     # position
            0.5,               # velocity
            0.2,               # yaw
            0.5                # yaw_rate
        ])

        # Process noise
        self.ukf.Q = np.diag([
            0.003, 0.003, 0.003,
            0.005,
            0.003,
            0.005
        ])

        # Measurement noise (vision)
        self.ukf.R = np.diag([
            0.10, 0.10, 0.15,   # position
            0.05                # yaw only
        ])

    def fx(self, x, dt):
        px, py, pz, v, yaw, yaw_rate = x

        # State propagation
        px += v * np.cos(yaw) * dt
        py += v * np.sin(yaw) * dt
        pz = pz
        yaw += yaw_rate * dt

        yaw = self._wrap_angle(yaw)

        return np.array([
            px,
            py,
            pz,
            v,
            yaw,
            yaw_rate
        ])

    # Measurement model
    def hx(self, x):
        px, py, pz, _, yaw, _ = x

        return np.array([
            px, py, pz,
            yaw     # roll & pitch unused
        ])

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi


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
