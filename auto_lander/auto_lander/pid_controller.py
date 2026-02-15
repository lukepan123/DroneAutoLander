#!/usr/bin/env python3
# https://www.iieta.org/journals/mmep/paper/10.18280/mmep.090607

import numpy as np

from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_euler

class PIDController:
    def __init__(self):
        # ---- STATE VARIABLES ----
        # PID Params
        self.dt = 0.5
        self.N = 40

    def pn_guidance(self, u, du, lam):
        """
        PN acceleration perpendicular to LOS.
        u  = LOS position vector (NED) = p_a - p_m
        du = LOS velocity vector (NED) = v_a - v_m
        lam = navigation constant (gain λ)

        Returns: a_perp (3D acceleration vector)
        """
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            return np.zeros(3)

        # LOS rotation vector Ω = (u × du) / (u·u)
        Omega = np.cross(u, du) / (u_norm**2)

        # PN command: a_perp = -λ * |du| * (u/|u| × Ω)
        a_perp = -lam * np.linalg.norm(du) * np.cross(u/u_norm, Omega)
        return a_perp
    
    def approach_accel(self, u, du, Kp, Kd):
        """
        LOS-aligned approach controller.
        a_parallel = Kp * u + Kd * du
        """
        return Kp * u + Kd * du
    
    def total_accel(self, u, du, lam, Kp, Kd):
        a_perp = self.pn_guidance(u, du, lam)
        a_parallel = self.approach_accel(u, du, Kp, Kd)
        a = a_perp + a_parallel

        # Paper: disregard vertical (z) component for horizontal guidance
        a[2] = 0
        return a
    
    def accel_to_attitude(self, ax, ay, vx, vy, m, g, kd):
        """
        Convert horizontal accelerations → desired roll (φ) and pitch (θ)
        using eq. (3) and (4).

        ax, ay = desired accelerations (N frame)
        vx, vy = current horizontal velocities
        kd = drag constant
        """
        # Signed quadratic drag terms
        drag_x = kd * vx * abs(vx)
        drag_y = kd * vy * abs(vy)

        # Pitch θ (nose down positive in NED)
        theta = -np.arctan((m*ax + drag_x) / (m*g))

        # Roll φ
        phi = np.arctan((np.cos(theta) * (m*ay + drag_y)) / (m*g))

        return phi, theta
    
    def compute_throttle(self, phi, theta, m, g):
        max_thrust = 48.0
        thrust = m * g / (np.cos(phi) * np.cos(theta))
        throttle = thrust/max_thrust
        return throttle
    
    def build_mavros_msg(self, phi, theta, yaw_des, thrust):
        """
        Returns: AttitudeTarget message for MAVROS /setpoint_raw/attitude
        """

        # MAVROS uses quaternion for attitude
        q = quaternion_from_euler(phi, theta, yaw_des)

        msg = AttitudeTarget()
        msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                        AttitudeTarget.IGNORE_PITCH_RATE | \
                        AttitudeTarget.IGNORE_YAW_RATE

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        msg.thrust = thrust

        return msg
    
    def pn_controller(self, p_a, v_a, p_m, v_m, yaw_des,
                    lam, Kp, Kd,
                    m=1.0, g=9.81, kd=0.1):
        """
        p_a, v_a = drone position & velocity (NED)
        p_m, v_m = target position & velocity (NED)
        yaw_des = desired yaw (point towards target)
        """

        u = p_a - p_m
        du = v_a - v_m

        # Step 1–3: PN + LOS-PD accelerations
        acc = self.total_accel(u, du, lam, Kp, Kd)

        # Step 4: convert accel → attitude
        phi, theta = self.accel_to_attitude(
            acc[0], acc[1],
            vx=v_a[0], vy=v_a[1],
            m=m, g=g, kd=kd
        )

        # Step 5: compute throttle
        T = self.compute_throttle(phi, theta, m, g)

        # dx = p_m[0] - p_a[0]
        # dy = p_m[1] - p_a[1]
        # yaw_des = np.arctan2(dy, dx)
        # yaw_des = (yaw_des + np.pi) % (2*np.pi) - np.pi # normalise

        # Step 6: build MAVROS message
        msg = self.build_mavros_msg(phi, theta, yaw_des, thrust=T)

        return msg
    
    def update(self, node):
        msg = self.pn_controller(
            p_a=np.array([node.odometry.pose.pose.position.x,
                          node.odometry.pose.pose.position.y,
                          node.odometry.pose.pose.position.z]),
            v_a=np.array([node.odometry.twist.twist.linear.x,
                          node.odometry.twist.twist.linear.y,
                          node.odometry.twist.twist.linear.z]),
            p_m=np.array(node.landing_pad_position),
            v_m=np.array(node.landing_pad_velocity),
            yaw_des=0.0,
            lam=4.5,
            Kp=4.0,
            Kd=3.0,
            m=1.98,
            g=9.81,
            kd=0.25
        )

        node.att_pub.publish(msg)

