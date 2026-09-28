import numpy as np
import math

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import GlobalPositionTarget
from mavros_msgs.msg import AttitudeTarget
from tf_transformations import quaternion_from_euler

from .state_definitions import QUAD_State

""" PD/PN Controller derived from https://www.professeurs.polymtl.ca/jerome.le-ny/docs/journals/2017_JGCD_MAVlanding.pdf
    Adjusted yaw and altitude control to allow for direct control.
"""

CONTROL_OUTPUT_KEYS = (
    # Commanded forces (N, ENU, after limiting) and resulting thrust magnitude
    "F_x", "F_y", "F_z", "F_thrust",
    # Commanded throttle / attitude (after clamps and slew limiting)
    "throttle_cmd", "roll_cmd", "pitch_cmd", "yaw_cmd",
    # Guidance terms (z is omitted for perp/par: the altitude loop overwrites it)
    "accel_perp_x", "accel_perp_y", "accel_par_x", "accel_par_y", "accel_z_cmd",
    "z_err", "vel_z_err", "vel_z_integral", "gain_factor", "lam",
    # Limiter flags (1 = active this tick)
    "sat_accel_z", "sat_vel_z_int", "sat_F_z", "sat_F_horiz",
    "sat_roll_clamp", "sat_pitch_clamp",
    "slew_throttle", "slew_roll", "slew_pitch", "slew_yaw",
    # Mode flags
    "marker_yaw_valid", "cutoff_active",
)

class PIDController:
    """ Defines the PID Controller Class
    """

    def __init__(self, dt):
        """ Initialise the PID Controller node
        """

        # ---- PID PARAMETERS ----
        # PN/PD Gains
        self.lam_0 = 2.0
        self.Kp_0 = 6.5
        self.Kd_0 = 3.25

        # P/PI Altitude Gains
        self.Kp_pos_z = 0.2
        self.Kp_vel_z = 2.0
        self.Ki_vel_z = 0.2

        self.vel_z_err = 0.0
        self.vel_z_integral = 0.0
        self.vel_z_i_clamp = 2.0  # m/s² — prevents windup

        # Constant Parameters
        self.m = 1.98
        self.max_thrust = 40.0
        self.g = 9.81
        self.cD = 0.002

        self.max_throttle_rate = 1.0   # unit/s
        self.max_angle_rate = 1.0      # rad/s
        self.prev_throttle = self.m * self.g / self.max_thrust
        self.prev_phi = 0.0
        self.prev_theta = 0.0
        self.prev_yaw =  None

        self.d_blend_start = 4.0   # start blending toward marker yaw
        self.d_blend_end   = 1.0   # fully aligned with marker yaw
        self.d_hold_radius = 4.0   # inside this, hold yaw if tag lost

        # ---- STATE VARIABLES ----
        self.lam = self.lam_0
        self.Kp = self.Kp_0
        self.Kd = self.Kd_0

        self.dt = dt

        # Logging
        self._last_outputs = self._nan_outputs()

    def controller(
        self,
        node,
        target_altitude,
        target_desc_rate,
        marker_yaw,
        cutoff,
        alt_pos_control,
        quad_yaw,
        quad_vel,
        u,
        du,
    ):
        """PN/PD Controller Logic

        :param target_altitude:  desired hover/approach altitude (m, ENU z)
        :param target_desc_rate: desired descent rate (m/s, ENU z)
        :param marker_yaw:       AprilTag-measured pad yaw (rad, ENU), or None/NaN
                                 if no tag is currently detected
        :param cutoff:           bool — if True, zero thrust and hold yaw (kill switch)
        :param alt_pos_control:  bool - if True, control altitude through position SP, false, control altitude rate
        :param quad_yaw:         drone yaw   ψ   (rad)
        :param quad_vel:         drone velocity  v_a (m/s) — 3-vector [vx, vy, vz]
        :param u:                target-drone relative position p_m (ENU globally aligned) (m) — 3-vector [x, y, z]
        :param du:               target-drone velocity v_m (m/s) (ENU globally aligned) — 3-vector [vx, vy, vz]
        :return: AttitudeTarget msg (attitude quaternion + normalised throttle)
        """
        tag_detected = (marker_yaw is not None) and not np.isnan(marker_yaw)

        if self.prev_yaw is None:
            self.prev_yaw = quad_yaw

        if cutoff:
            # kill switch — skip blend/hold logic entirely, just freeze yaw
            target_yaw = self.prev_yaw
        else:
            dist_to_pad = np.hypot(u[0], u[1])        # horizontal range to pad
            heading_to_pad = np.arctan2(u[1], u[0])   # bearing drone -> pad

            if tag_detected:
                if dist_to_pad > self.d_blend_start:
                    target_yaw = heading_to_pad
                elif dist_to_pad > self.d_blend_end:
                    alpha = (self.d_blend_start - dist_to_pad) / (self.d_blend_start - self.d_blend_end)
                    target_yaw = self._blend_angle(heading_to_pad, marker_yaw, alpha)
                else:
                    target_yaw = marker_yaw
            else:
                if dist_to_pad < self.d_hold_radius:
                    target_yaw = self.prev_yaw    # hold — don't chase a noisy/unavailable bearing near the pad
                else:
                    target_yaw = heading_to_pad   # still far out, LOS heading is fine

        # Condition yaw
        target_yaw = (target_yaw + np.pi) % (2 * np.pi) - np.pi #type: ignore
        yaw_pre_slew = target_yaw
        target_yaw = self._slew_angle(target_yaw, self.prev_yaw, self.max_angle_rate)
        slew_yaw = abs((target_yaw - yaw_pre_slew + np.pi) % (2 * np.pi) - np.pi) > 1e-9
        self.prev_yaw = target_yaw

        # Cuttoff condition
        if cutoff is True:
            # ---- Build MAVROS message (ENU)
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

            # Logging
            out = self._nan_outputs()
            out.update(
                F_x=0.0, F_y=0.0, F_z=0.0, F_thrust=0.0,
                throttle_cmd=0.0, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=float(target_yaw),
                slew_yaw=int(slew_yaw), marker_yaw_valid=int(tag_detected), cutoff_active=1,
            )
            self._last_outputs = out

            return msg

        # ---- PN/PD Controller (ENU) ----
        r = np.linalg.norm(u[:2])
        drop_off_strength = 0.5
        lam_gain_factor = 1 - np.exp(-drop_off_strength * r)

        terminal_gain = 1.0
        drop_off = 7.0

        u_norm = np.linalg.norm(u)
        gain_factor = terminal_gain * drop_off / (u_norm**2 + drop_off)

        self.lam = self.lam_0 * lam_gain_factor
        self.Kp = self.Kp_0 * gain_factor
        self.Kd = self.Kd_0 * gain_factor

        if u_norm < 1e-6:
            accel_perp = np.zeros(3)
        else:
            omega = np.cross(u, du) / (u_norm**2)
            accel_perp = -self.lam * np.linalg.norm(du) * np.cross(u / u_norm, omega)

        accel_parallel = self.Kp * u + self.Kd * du
        accel = accel_perp + accel_parallel

        # ---- Altitude Controller ----
        z_err = target_altitude - (-u[QUAD_State.Z])

        if alt_pos_control:
            vel_z_err = np.clip(self.Kp_pos_z * z_err, -1.5, 1.5) + du[QUAD_State.Z]
        else:
            vel_z_err = target_desc_rate + du[QUAD_State.Z]

        integral_raw = self.vel_z_integral + vel_z_err * self.dt
        self.vel_z_integral = np.clip(integral_raw, -self.vel_z_i_clamp, self.vel_z_i_clamp)
        sat_vel_z_int = abs(integral_raw) > self.vel_z_i_clamp

        accel_z_raw = self.Kp_vel_z * vel_z_err + self.Ki_vel_z * self.vel_z_integral
        accel[QUAD_State.Z] = np.clip(accel_z_raw, -1.0 * self.g, 1.0 * self.g)
        sat_accel_z = abs(accel_z_raw) > self.g

        # --- Final Output ---
        drag_x = self.cD * quad_vel[QUAD_State.X] * abs(quad_vel[QUAD_State.X])
        drag_y = self.cD * quad_vel[QUAD_State.Y] * abs(quad_vel[QUAD_State.Y])
        drag_z = self.cD * quad_vel[QUAD_State.Z] * abs(quad_vel[QUAD_State.Z])

        F_x = self.m * accel[QUAD_State.X] + drag_x
        F_y = self.m * accel[QUAD_State.Y] + drag_y
        F_z_raw = self.m * (accel[QUAD_State.Z] + self.g) + drag_z
        F_z = np.clip(F_z_raw, 0.0, self.max_thrust)
        sat_F_z = F_z_raw < 0.0 or F_z_raw > self.max_thrust

        horiz_budget = np.sqrt(max(self.max_thrust**2 - F_z**2, 0.0))
        F_horiz = np.hypot(F_x, F_y)
        sat_F_horiz = F_horiz > horiz_budget
        if sat_F_horiz:
            scale = horiz_budget / F_horiz
            F_x *= scale
            F_y *= scale

        thrust = np.sqrt(F_x**2 + F_y**2 + F_z**2)

        sat_roll = sat_pitch = False
        if thrust < 1e-6:
            phi = self.prev_phi
            theta = self.prev_theta
            throttle = 0.0
        else:
            phi = np.arcsin((F_x * np.sin(quad_yaw) - F_y * np.cos(quad_yaw)) / thrust)
            sat_roll = abs(phi) > 1.0
            phi = max(-1, min(phi, 1))

            theta = np.arctan2((F_x * np.cos(quad_yaw) + F_y * np.sin(quad_yaw)), F_z)
            sat_pitch = abs(theta) > 1.0
            theta = max(-1, min(theta, 1))

            throttle = np.clip(thrust / self.max_thrust, 0.0, 1.0)

        # Restrict/ramp outputs
        throttle_pre, phi_pre, theta_pre = throttle, phi, theta
        throttle = self._slew(throttle, self.prev_throttle, self.max_throttle_rate)
        phi = self._slew(phi, self.prev_phi, self.max_angle_rate)
        theta = self._slew(theta, self.prev_theta, self.max_angle_rate)
        self.prev_throttle, self.prev_phi, self.prev_theta = throttle, phi, theta

        # ---- Stash outputs for logging ----
        out = self._nan_outputs()
        out.update(
            F_x=float(F_x), F_y=float(F_y), F_z=float(F_z), F_thrust=float(thrust),
            throttle_cmd=float(throttle), roll_cmd=float(phi), pitch_cmd=float(theta),
            yaw_cmd=float(target_yaw),
            accel_perp_x=float(accel_perp[0]), accel_perp_y=float(accel_perp[1]),
            accel_par_x=float(accel_parallel[0]), accel_par_y=float(accel_parallel[1]),
            accel_z_cmd=float(accel[QUAD_State.Z]),
            z_err=float(z_err), vel_z_err=float(vel_z_err),
            vel_z_integral=float(self.vel_z_integral),
            gain_factor=float(gain_factor), lam=float(self.lam),
            sat_accel_z=int(sat_accel_z), sat_vel_z_int=int(sat_vel_z_int),
            sat_F_z=int(sat_F_z), sat_F_horiz=int(sat_F_horiz),
            sat_roll_clamp=int(sat_roll), sat_pitch_clamp=int(sat_pitch),
            slew_throttle=int(abs(throttle - throttle_pre) > 1e-9),
            slew_roll=int(abs(phi - phi_pre) > 1e-9),
            slew_pitch=int(abs(theta - theta_pre) > 1e-9),
            slew_yaw=int(slew_yaw),
            marker_yaw_valid=int(tag_detected),   # passed the sigma_yaw gate in update()
            cutoff_active=0,
        )
        self._last_outputs = out

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
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.1

        # Drive outputs to quadcopter via MAVROS
        node.vel_pub.publish(msg)


    def update(self, node):
        """ Update the PID controller with current state to generate new controller 
            commands.
        """

        # Only pass through marker yaw if uncertainty on yaw is low enough
        if node._UKF_diag["sigma_yaw"] <= 0.15:
            marker_yaw = node.landing_pad_yaw_forward_predict
        else:
            marker_yaw = None 

        msg = self.controller(
            node=node,
            target_altitude=node.target_z,
            target_desc_rate=node.target_z_rate,
            marker_yaw=marker_yaw,
            cutoff=node.cutoff,
            alt_pos_control=node.alt_pos_control,
            quad_yaw=node.quad_yaw,
            quad_vel=np.array(
                [
                    node.odometry.twist.twist.linear.x,
                    node.odometry.twist.twist.linear.y,
                    node.odometry.twist.twist.linear.z,
                ]
            ),
            u=np.array(node.landing_pad_relative_position_forward_predict),
            du=np.array(node.landing_pad_relative_velocity_forward_predict),
        )

        # Drive outputs to quadcopter via MAVROS
        node.att_pub.publish(msg)


    def goPosition(self, node, lat, lon, alt, yaw=None):
        """ Command the drone to fly to a global GPS position via MAVROS.
    
            lat, lon : degrees (WGS84)
            alt      : meters ABOVE HOME (FRAME_GLOBAL_REL_ALT)
            yaw      : optional heading in radians, ENU convention
                    (0 = east, pi/2 = north). If None, yaw is ignored and
                    the FCU holds its current heading.
    
            Must be called continuously (>2 Hz) while in GUIDED/OFFBOARD mode.
        """
    
        msg = GlobalPositionTarget()
        msg.header.stamp = node.get_clock().now().to_msg()  # ROS1: rospy.Time.now()
        msg.header.frame_id = "map"
    
        msg.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
    
        # Use position (+ yaw if given); ignore velocity, accel, yaw rate
        msg.type_mask = (
            GlobalPositionTarget.IGNORE_VX
            | GlobalPositionTarget.IGNORE_VY
            | GlobalPositionTarget.IGNORE_VZ
            | GlobalPositionTarget.IGNORE_AFX
            | GlobalPositionTarget.IGNORE_AFY
            | GlobalPositionTarget.IGNORE_AFZ
            | GlobalPositionTarget.IGNORE_YAW_RATE
        )
    
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
    
        if yaw is None:
            msg.type_mask |= GlobalPositionTarget.IGNORE_YAW
        else:
            msg.yaw = yaw
    
        # Drive output to quadcopter via MAVROS
        node.global_pos_pub.publish(msg)


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


    @staticmethod
    def _blend_angle(a, b, alpha):
        """Interpolate from angle a to angle b via unit vectors (shortest-arc, wrap-safe)."""
        v = (1 - alpha) * np.array([np.cos(a), np.sin(a)]) + alpha * np.array([np.cos(b), np.sin(b)])
        return np.arctan2(v[1], v[0])

    # Logging functions
    def _nan_outputs(self) -> dict:
        return {k: float("nan") for k in CONTROL_OUTPUT_KEYS}

    
    def reset_outputs(self) -> None:
        """Call once per control tick before the controller may run, so ticks
        where controller() isn't called log NaN instead of stale values."""
        self._last_outputs = {k: float("nan") for k in CONTROL_OUTPUT_KEYS}


    def get_control_outputs(self) -> dict:
        """Most recent controller outputs and limiter flags (copy)."""
        return dict(self._last_outputs)
