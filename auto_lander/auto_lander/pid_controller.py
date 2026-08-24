import numpy as np

from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import AttitudeTarget
from tf_transformations import quaternion_from_euler

from .state_definitions import QUAD_State

""" PD/PN Controller derived from https://www.professeurs.polymtl.ca/jerome.le-ny/docs/journals/2017_JGCD_MAVlanding.pdf
    Adjusted yaw and altitude control to allow for direct control.
"""

class PIDController:
    """ Defines the PID Controller Class
    """

    def __init__(self, dt):
        """ Initialise the PID Controller node
        """

        # ---- PID PARAMETERS ----
        # PN/PD Gains
        self.lam_0 = 2.0
        self.Kp_0 = 6.0
        self.Kd_0 = 3.0

        # P/PI Altitude Gains
        self.Kp_z_pos = 0.2

        self.Kp_vel_z = 5.0
        self.Ki_vel_z = 2.0

        self.vel_z_err = 0.0
        self.vel_z_integral = 0.0
        self.vel_z_i_clamp = 2.0  # m/s² — prevents windup

        # Constant Parameters
        self.m = 1.98
        self.max_thrust = 46.0
        self.g = 9.81
        self.cD = -0.002 #TODO: I believe the current velocity and acceleration for the 
                         #      quad are negated, and hence is why a negative CD gives 
                         #      the correct effect... will fix later.

        self.max_throttle_rate = 0.4   # unit/s
        self.max_angle_rate = 1.0      # rad/s
        self.prev_throttle = 0.0
        self.prev_phi = 0.0
        self.prev_theta = 0.0
        self.prev_yaw = 1.5707963


        # ---- STATE VARIABLES ----
        self.lam = self.lam_0
        self.Kp = self.Kp_0
        self.Kd = self.Kd_0

        self.dt = dt

    def controller(
        self,
        target_altitude,
        target_yaw,
        cutoff,
        quad_yaw,
        quad_vel,
        u,
        du,
    ):
        """PN/PD Controller Logic

        :param target_altitude: desired hover/approach altitude (m, ENU z)
        :param target_yaw:      desired heading (rad, ENU convention)
        :param cutoff:          bool — if True, zero thrust and hold yaw (kill switch)
        :param quad_yaw:        drone yaw   ψ   (rad)
        :param quad_vel:        drone velocity  v_a (m/s) — 3-vector [vx, vy, vz]
        :param u:               drone-target relative position p_m (globally aligned) (m) — 3-vector [x, y, z]
        :param du:              drone-target velocity v_m (m/s) (globally aligned) — 3-vector [vx, vy, vz]
        :return: AttitudeTarget msg (attitude quaternion + normalised throttle)
        """

        # Condition yaw
        target_yaw = (target_yaw + np.pi) % (2 * np.pi) - np.pi
        target_yaw = self._slew_angle(target_yaw, self.prev_yaw, self.max_angle_rate)
        self.prev_yaw = target_yaw

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

        # ---- PN/PD Controller ----
        # Drop off bearing/PN gain as we close to target
        r = np.linalg.norm(u[:2])
        drop_off_strength = 0.5
        lam_gain_factor = 1 - np.exp(-drop_off_strength * r)

        terminal_gain = 1.0
        drop_off = 5.0
        peak_gain_dist = 0.0

        u_norm = np.linalg.norm(u)
        gain_factor = terminal_gain * drop_off / ((u_norm - peak_gain_dist)**2 + drop_off)

        self.lam = self.lam_0 * lam_gain_factor
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

        # ---- Altitude Controller ----
        # Outer P loop: position error → velocity command
        z_err = u[QUAD_State.Z] - target_altitude
        vel_z_des = np.clip(self.Kp_z_pos * z_err, -1.5, 1.5)

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

        # --- Final Output ---
        # Signed quadratic drag terms
        drag_x = self.cD * quad_vel[QUAD_State.X] * abs(quad_vel[QUAD_State.X])
        drag_y = self.cD * quad_vel[QUAD_State.Y] * abs(quad_vel[QUAD_State.Y])
        drag_z = self.cD * quad_vel[QUAD_State.Z] * abs(quad_vel[QUAD_State.Z])

        # Rewrite in terms of forces
        F_x = self.m * accel[QUAD_State.X] + drag_x
        F_y = self.m * accel[QUAD_State.Y] + drag_y
        F_z = self.m * (accel[QUAD_State.Z] - self.g) + drag_z

        # Thrust/Throttle
        thrust = np.sqrt(F_x**2 + F_y**2 + F_z**2)
        throttle = np.clip(thrust / self.max_thrust, 0.0, 1.0)

        # Roll φ
        phi = np.arcsin(-(F_x * np.sin(quad_yaw) - F_y * np.cos(quad_yaw)) / (thrust))
        phi = max(-1, min(phi, 1))

        # Pitch θ (nose down positive in NED)
        theta = np.arcsin(
            -(F_x * np.cos(quad_yaw) + F_y * np.sin(quad_yaw)) / (thrust * np.cos(phi))
        )
        theta = max(-1, min(theta, 1))

        # Restrict/ramp outputs
        throttle = self._slew(throttle, self.prev_throttle, self.max_throttle_rate)
        phi = self._slew(phi, self.prev_phi, self.max_angle_rate)
        theta = self._slew(theta, self.prev_theta, self.max_angle_rate)
        self.prev_throttle, self.prev_phi, self.prev_theta = throttle, phi, theta

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


    def stop(self, node):
        """ Makes drone stop moving
        """

        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        # linear/angular default to 0.0, but explicit for clarity
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        # Drive outputs to quadcopter via MAVROS
        node.vel_pub.publish(msg)


    def look(self, node):
        """ Makes drone yaw to search for target
        """

        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        # linear/angular default to 0.0, but explicit for clarity
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.1
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        # Drive outputs to quadcopter via MAVROS
        node.vel_pub.publish(msg)


    def update(self, node):
        """ Update the PID controller with current state to generate new controller 
            commands.
        """
        msg = self.controller(
            target_altitude=node.target_z,
            target_yaw=node.landing_pad_yaw_forward_predict,
            cutoff=node.cutoff,
            quad_yaw=node.quad_yaw,
            quad_vel=np.array(
                [
                    node.odometry.twist.twist.linear.x,
                    node.odometry.twist.twist.linear.y,
                    node.odometry.twist.twist.linear.z,
                ]
            ),
            u=-np.array(node.landing_pad_relative_position_forward_predict),
            du=-np.array(node.landing_pad_relative_velocity_forward_predict),
        )

        # Drive outputs to quadcopter via MAVROS
        node.att_pub.publish(msg)


    def _slew(self, target, prev, max_rate):
        """ Slew the PID output to the maximum rate.

        :param target: Target output
        :param prev: Previous output value
        :param max_rate: Maximum rate of change in output
        :return: Slewed output
        """
        max_step = max_rate * self.dt
        return prev + np.clip(target - prev, -max_step, max_step)


    def _slew_angle(self, target, prev, max_rate):
        """ Slew the PID angle output to the maximum rate.

        :param target: Target output
        :param prev: Previous output value
        :param max_rate: Maximum rate of change in output
        :return: Slewed output
        """
        max_step = max_rate * self.dt
        diff = (target - prev + np.pi) % (2 * np.pi) - np.pi
        result = prev + np.clip(diff, -max_step, max_step)
        return (result + np.pi) % (2 * np.pi) - np.pi
