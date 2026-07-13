#!/usr/bin/env python3
# https://www.professeurs.polymtl.ca/jerome.le-ny/docs/journals/2017_JGCD_MAVlanding.pdf

import numpy as np

from mavros_msgs.msg import AttitudeTarget
from tf_transformations import quaternion_from_euler


class PIDController:
    def __init__(self, dt):
        # ---- STATE VARIABLES ----
        # PN/PD Gains
        self.lam_0 = 8.0
        self.Kp_0 = 7.0
        self.Kd_0 = 4.0

        # P/PI Altitude Gains
        self.Kp_z_pos = 0.1

        self.Kp_vel_z = 5.0
        self.Ki_vel_z = 2.0

        self.vel_z_err = 0.0
        self.vel_z_integral = 0.0
        self.vel_z_i_clamp = 2.0  # m/s² — prevents windup

        # Constants
        self.m = 1.98
        self.max_thrust = 46.0
        self.g = 9.81
        self.cD = -0.002

        # Initialise variables
        self.lam = self.lam_0
        self.Kp = self.Kp_0
        self.Kd = self.Kd_0

        self.dt = dt

    def controller(
        self,
        target_altitude,
        target_yaw,
        cutoff,
        quad_roll,
        quad_pitch,
        quad_yaw,
        quad_vel,
        u,
        du,
    ):
        """PN/PD Controller Logic

        :param target_altitude: desired hover/approach altitude (m, ENU z)
        :param target_yaw:      desired heading (rad, ENU convention)
        :param cutoff:          bool — if True, zero thrust and hold yaw (kill switch)
        :param quad_roll:       drone roll  φ   (rad)
        :param quad_pitch:      drone pitch θ   (rad)
        :param quad_yaw:        drone yaw   ψ   (rad)
        :param quad_vel:        drone velocity  v_a (m/s) — 3-vector [vx, vy, vz]
        :param u:               drone-target relative position p_m (globally aligned) (m) — 3-vector [x, y, z]
        :param du:              drone-target velocity v_m (m/s) (globally aligned) — 3-vector [vx, vy, vz]
        :return: AttitudeTarget msg (attitude quaternion + normalised throttle)
        """

        # Condition yaw
        target_yaw = (target_yaw + np.pi) % (2 * np.pi) - np.pi

        # Cuttoff condition
        if cutoff is True:
            # ---- Build MAVROS message
            q = quaternion_from_euler(0, 0, target_yaw)

            msg = AttitudeTarget()
            msg.type_mask = (
                AttitudeTarget.IGNORE_ROLL_RATE
                | AttitudeTarget.IGNORE_PITCH_RATE
                | AttitudeTarget.IGNORE_YAW_RATE
            )

            msg.orientation.x = q[0]
            msg.orientation.y = q[1]
            msg.orientation.z = q[2]
            msg.orientation.w = q[3]
            msg.thrust = 0.0

            return msg

        # ======================
        # == PN/PD Controller ==
        # ======================

        # Increase gains as distance to target decreases
        terminal_gain = 0.9
        no_gain_dist = 2.0

        u_norm = np.linalg.norm(u)
        gain_factor = terminal_gain * no_gain_dist / (u_norm + no_gain_dist)

        self.lam = self.lam_0 * gain_factor
        self.Kp = self.Kp_0 * gain_factor
        self.Kd = self.Kd_0 * gain_factor

        # Calculate PN acceleration
        if u_norm < 1e-6:
            accel_perp = np.zeros(3)
        else:
            # LOS rotation vector Ω = (u × du) / (u·u)
            omega = np.cross(u, du) / (u_norm**2)

            # PN command: a_perp = -λ * |du| * (u/|u| × Ω)
            accel_perp = -self.lam * np.linalg.norm(du) * np.cross(u / u_norm, omega)

        # Calculate LOS PD acceleration
        accel_parallel = self.Kp * u + self.Kd * du

        # Sum acceleration
        accel = accel_perp + accel_parallel

        # =========================
        # = Altitude Controller =
        # =========================

        # Outer P loop: position error → velocity command
        z_err = u[2] - target_altitude
        vel_z_des = np.clip(self.Kp_z_pos * z_err, -0.5, 0.5)

        # Inner PI loop: velocity error → acceleration command
        vel_z_err = vel_z_des - quad_vel[2]
        self.vel_z_integral = np.clip(
            self.vel_z_integral + vel_z_err * self.dt,
            -self.vel_z_i_clamp,
            self.vel_z_i_clamp,
        )
        accel[2] = np.clip(
            self.Kp_vel_z * vel_z_err + self.Ki_vel_z * self.vel_z_integral,
            -1.0 * self.g,
            1.0 * self.g,
        )

        # ================================
        # == Combine controller outputs ==
        # ================================

        # ---- Convert linear accel into attitudes
        # Signed quadratic drag terms
        drag_x = self.cD * quad_vel[0] * abs(quad_vel[0])
        drag_y = self.cD * quad_vel[1] * abs(quad_vel[1])
        drag_z = self.cD * quad_vel[2] * abs(quad_vel[2])

        # Rewrite in terms of forces
        F_x = self.m * accel[0] + drag_x
        F_y = self.m * accel[1] + drag_y
        F_z = self.m * (accel[2] - self.g) + drag_z

        # Thrust/Throttle
        thrust = np.sqrt(F_x**2 + F_y**2 + F_z**2)
        throttle = np.clip(thrust / self.max_thrust, 0.0, 1.0)

        # Roll φ
        # phi = np.arctan((np.cos(theta) * (self.m*accel[1] + drag_y)) / (self.m*self.g))
        phi = np.arcsin(-(F_x * np.sin(quad_yaw) - F_y * np.cos(quad_yaw)) / (thrust))
        phi = max(-1, min(phi, 1))

        # Pitch θ (nose down positive in NED)
        # theta = -np.arctan((self.m*accel[0] + drag_x) / (self.m*self.g))
        theta = np.arcsin(
            -(F_x * np.cos(quad_yaw) + F_y * np.sin(quad_yaw)) / (thrust * np.cos(phi))
        )
        theta = max(-1, min(theta, 1))

        # ---- Build MAVROS message
        q = quaternion_from_euler(phi, theta, target_yaw)

        msg = AttitudeTarget()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        msg.thrust = throttle

        return msg

    def update(self, node):
        msg = self.controller(
            target_altitude=node.target_z,
            target_yaw=node.landing_pad_yaw,
            cutoff=node.cutoff,
            quad_roll=node.roll,
            quad_pitch=node.pitch,
            quad_yaw=node.yaw,
            quad_vel=np.array(
                [
                    node.odometry.twist.twist.linear.x,
                    node.odometry.twist.twist.linear.y,
                    node.odometry.twist.twist.linear.z,
                ]
            ),
            u=-np.array(node.landing_pad_relative_position),
            du=-np.array(node.landing_pad_relative_velocity),
        )

        node.att_pub.publish(msg)

    def stop(self, node):
        msg = self.controller(
            target_altitude=node.target_z,
            target_yaw=node.landing_pad_yaw,
            cutoff=node.cutoff,
            quad_roll=node.roll,
            quad_pitch=node.pitch,
            quad_yaw=node.yaw,
            quad_vel=np.array(
                [
                    node.odometry.twist.twist.linear.x,
                    node.odometry.twist.twist.linear.y,
                    node.odometry.twist.twist.linear.z,
                ]
            ),
            u=-np.array([0, 0, 0]),
            du=-np.array([0, 0, 0]),
        )

        node.att_pub.publish(msg)
