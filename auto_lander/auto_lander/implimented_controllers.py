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

        # Position error
        x_error = node.target_pose.pose.position.x - node.pose.pose.position.x
        y_error = node.target_pose.pose.position.y - node.pose.pose.position.y

        # Integrate error
        node.x_error_int += x_error * dt
        node.y_error_int += y_error * dt

        # Anti-windup clamp
        max_int = 2.0
        node.x_error_int = max(min(node.x_error_int, max_int), -max_int)
        node.y_error_int = max(min(node.y_error_int, max_int), -max_int)

        # Derivative
        dx_meas = (node.pose.pose.position.x - node.prev_x_meas) / dt
        dy_meas = (node.pose.pose.position.y - node.prev_y_meas) / dt

        node.prev_x_meas = node.pose.pose.position.x
        node.prev_y_meas = node.pose.pose.position.y

        # PID output
        vx = node.kp * x_error + node.ki * node.x_error_int + node.kd * dx_meas
        vy = node.kp * y_error + node.ki * node.y_error_int + node.kd * dy_meas

        # Velocity saturation
        max_vel = 2.0
        vx = max(min(vx, max_vel), -max_vel)
        vy = max(min(vy, max_vel), -max_vel)

        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
            
        node.vel_pub.publish(msg)

        print("X_drone", node.pose.pose.position.x, "Y_drone", node.pose.pose.position.y, "Z_drone", node.pose.pose.position.z)
        print("X_targ", node.target_pose.pose.position.x, "Y_targ", node.target_pose.pose.position.y, "Z_targ", node.target_pose.pose.position.z)
        print("X_err", x_error, "Y_err", y_error, "Z_targ")
        print("X_com", vx, "Y_com", vy)
        #error_mag = np.sqrt(np.power(x_error, 2) + np.power(y_error, 2))
        #print("Distance Error: ", error_mag)


    return msg
