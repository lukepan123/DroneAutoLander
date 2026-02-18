#!/usr/bin/env python3
# https://www.iieta.org/journals/mmep/paper/10.18280/mmep.090607

import numpy as np

from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_euler

class PIDController:
    def __init__(self):
        # ---- STATE VARIABLES ----
        self.lam = 3.0
        self.Kp  = 3.0
        self.Kd  = 3.0
        self.cD  = 0.10

        self.m = 1.98
        self.max_thrust = 48.0
        self.g = 9.81
        self.cD = 0.1
    
    def pn_controller(self, quad_pos, quad_vel, landing_pad_pos, landing_pad_vel, quad_yaw):
        """
        p_a, v_a = drone position & velocity
        p_m, v_m = target position & velocity
        yaw_des = desired yaw (point towards target)
        """

        # ---- Calculate error vectors
        u = quad_pos - landing_pad_pos
        du = quad_vel - landing_pad_vel

        # ---- Calculate PN acceleration
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            accel_perp = np.zeros(3)
        else:
            # LOS rotation vector Ω = (u × du) / (u·u)
            omega = np.cross(u, du) / (u_norm**2)

            # PN command: a_perp = -λ * |du| * (u/|u| × Ω)
            accel_perp = -self.lam * np.linalg.norm(du) * np.cross(u/u_norm, omega)

        # ---- Calculate LOS PD acceleration
        accel_parallel = self.Kp * u + self.Kd * du

        # ---- Sum acceleration 
        accel = accel_perp + accel_parallel
        accel[2] = 0 # Paper: disregard vertical (z) component for horizontal guidance

        # ---- Convert linear accel into attitudes
        # Signed quadratic drag terms
        drag_x = self.cD * quad_vel[0] * abs(quad_vel[0])
        drag_y = self.cD * quad_vel[1] * abs(quad_vel[1])

        # Pitch θ (nose down positive in NED)
        theta = -np.arctan((self.m*accel[0] + drag_x) / (self.m*self.g))

        # Roll φ
        phi = np.arctan((np.cos(theta) * (self.m*accel[1] + drag_y)) / (self.m*self.g))

        # ---- Compute throttle
        thrust = self.m * self.g / (np.cos(phi) * np.cos(theta))
        throttle = thrust/self.max_thrust

        # ---- Build MAVROS message
        q = quaternion_from_euler(phi, theta, quad_yaw)

        msg = AttitudeTarget()
        msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                        AttitudeTarget.IGNORE_PITCH_RATE | \
                        AttitudeTarget.IGNORE_YAW_RATE

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        msg.thrust = throttle

        return msg
    
    def update(self, node):
        msg = self.pn_controller(
            quad_pos=np.array([node.odometry.pose.pose.position.x,
                          node.odometry.pose.pose.position.y,
                          node.odometry.pose.pose.position.z]),
            quad_vel=np.array([node.odometry.twist.twist.linear.x,
                          node.odometry.twist.twist.linear.y,
                          node.odometry.twist.twist.linear.z]),
            landing_pad_pos=np.array(node.landing_pad_position),
            landing_pad_vel=np.array(node.landing_pad_velocity),
            quad_yaw=0.0
        )

        node.att_pub.publish(msg)

