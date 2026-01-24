#!/usr/bin/env python3
import math
import numpy as np
from geometry_msgs.msg import TwistStamped

def Controller(node):
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'base_link'
           
    if node._tracking_enabled:
        # Time step
        now = node.get_clock().now()
        dt = (now - node.prev_time).nanoseconds * 1e-9
        dt = max(dt, 1e-3)
        node.prev_time = now

        # Determine target coordinates based on target position and velcoity
        lookout_factor = 1.0

        x_target = lookout_factor * node.target_vel[0] + node.target_pose[0]
        y_target = lookout_factor * node.target_vel[1] + node.target_pose[1]
        z_target = node.target_odometry.pose.pose.position.z + 0.5

        # node.get_logger().info(f'x {node.target_vel[0]}')
        # node.get_logger().info(f'y {node.target_vel[1]}')

        # Position error
        x_error = x_target - node.pose.pose.position.x
        y_error = y_target - node.pose.pose.position.y
        z_error = z_target - node.pose.pose.position.z

        # Integrate error
        node.x_error_int += x_error * dt
        node.y_error_int += y_error * dt

        # Anti-windup clamp
        max_int = 2.0
        node.x_error_int = max(min(node.x_error_int, max_int), -max_int)
        node.y_error_int = max(min(node.y_error_int, max_int), -max_int)

        # Derivative
        dx_error = (node.target_vel[0] - node.twist.twist.linear.x)
        dy_error = (node.target_vel[1] - node.twist.twist.linear.y)
        dz_error = (node.target_vel[2] - node.twist.twist.linear.z)

        node.x_p = node.kp * x_error
        node.x_i = node.ki * node.x_error_int
        node.x_d = node.kd * dx_error
        node.y_p = node.kp * y_error
        node.y_i = node.ki * node.y_error_int
        node.y_d = node.kd * dy_error

        # Make gains more sensitive as drone gets closer to rover
        node.kp = max(min((node.target_altitude + z_error)/7 * 2.5,1.5),0.2)

        # PID output
        vx = node.kp * x_error + node.ki * node.x_error_int + node.kd * dx_error
        vy = node.kp * y_error + node.ki * node.y_error_int + node.kd * dy_error
        vz = -0.3 # 0.04 * z_error

        # Velocity saturation
        max_vel = 8.0
        vx = max(min(vx, max_vel), -max_vel)
        vy = max(min(vy, max_vel), -max_vel)
        vz = max(min(vz, max_vel), -max_vel)

        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
            
        node.vel_pub.publish(msg)

        # print("X_drone", node.pose.pose.position.x, "Y_drone", node.pose.pose.position.y, "Z_drone", node.pose.pose.position.z)
        # print("X_targ", node.target_odometry.pose.pose.position.x, "Y_targ", node.target_odometry.pose.pose.position.y, "Z_targ", node.target_odometry.pose.pose.position.z)
        # print("X_err", x_error, "Y_err", y_error, "Z_targ")
        # print("X_com", vx, "Y_com", vy)
        #error_mag = np.sqrt(np.power(x_error, 2) + np.power(y_error, 2))
        #print("Distance Error: ", error_mag)

    return msg
