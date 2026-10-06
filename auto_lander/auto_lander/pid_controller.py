import numpy as np

from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import GlobalPositionTarget
from mavros_msgs.msg import AttitudeTarget
from tf_transformations import quaternion_from_euler

from .state_definitions import QUAD_State

""" PD/PN Controller derived from https://www.professeurs.polymtl.ca/jerome.le-ny/docs/journals/2017_JGCD_MAVlanding.pdf
    Adjusted yaw and altitude control to allow for direct control.
"""

CONTROL_OUTPUT_KEYS = (
    # Horizontal force terms (N, ENU), vertical reference force (m*g) and resulting magnitude
    "F_x", "F_y", "F_z", "F_thrust",
    # Commanded climb rate / thrust field / attitude (after clamps and slew limiting)
    "climb_rate_cmd", "thrust_field", "roll_cmd", "pitch_cmd", "yaw_cmd",
    # Guidance terms (z is omitted: the altitude loop handles it)
    "accel_perp_x", "accel_perp_y", "accel_par_x", "accel_par_y",
    "z_err", "gain_factor", "lam",
    # Limiter flags (1 = active this tick)
    "sat_climb_cmd", "sat_roll_clamp", "sat_pitch_clamp",
    "slew_roll", "slew_pitch", "slew_yaw",
    # Mode flags
    "marker_yaw_valid", "cutoff_active",
)

class PIDController:
    """Defines the PD/PN controller class."""

    def __init__(
        self,
        lam_0,
        Kp_0,
        Kd_0,
        Kp_pos_z,
        m,
        g,
        cD,
        max_angle_rate,
        d_blend_start,
        d_blend_end,
        d_hold_radius,
        drop_off_strength,
        terminal_gain,
        drop_off,
        vz_up_max,
        vz_dn_max,
        max_tilt=0.5,
        platform_vz_ff=False,
    ):
        """Initialise the PD/PN controller.

        :param max_climb_rate:  m/s, fallback for vz_up_max / vz_dn_max if those are not given
        :param vz_up_max:       m/s, MUST equal WPNAV_SPEED_UP / 100 on the FCU
        :param vz_dn_max:       m/s, MUST equal WPNAV_SPEED_DN / 100 on the FCU
        :param max_tilt:        rad, roll/pitch clamp, keep <= ANGLE_MAX on the FCU
        :param platform_vz_ff:  add the pad's vertical velocity as feedforward (moving platform)
        """
        # ---- PID PARAMETERS ----
        # PN/PD Gains
        self.lam_0 = lam_0
        self.Kp_0 = Kp_0
        self.Kd_0 = Kd_0

        # Altitude gain (outer height loop only; FCU does the rest)
        self.Kp_pos_z = Kp_pos_z

        # Constant Parameters
        self.m = m
        self.g = g
        self.cD = cD

        # Climb-rate limits (must match the FCU's WPNAV_SPEED_UP / WPNAV_SPEED_DN)
        self.vz_up_max = vz_up_max # m/s
        self.vz_dn_max = vz_dn_max # m/s
        self.max_tilt = max_tilt   # rad
        self.platform_vz_ff = platform_vz_ff

        self.max_angle_rate = max_angle_rate   # rad/s
        self.prev_phi = 0.0
        self.prev_theta = 0.0
        self.prev_yaw = None

        self.d_blend_start = d_blend_start   # start blending toward marker yaw
        self.d_blend_end = d_blend_end       # fully aligned with marker yaw
        self.d_hold_radius = d_hold_radius   # inside this, hold yaw if tag lost

        # Guidance tuning
        self.drop_off_strength = drop_off_strength
        self.terminal_gain = terminal_gain
        self.drop_off = drop_off

        # ---- STATE VARIABLES ----
        self.lam = self.lam_0
        self.Kp = self.Kp_0
        self.Kd = self.Kd_0

        # Logging
        self._last_outputs = self._nan_outputs()

    def controller(
        self,
        dt,
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
        """PD/PN Controller Logic

        :param target_altitude:  desired hover/approach altitude above the pad (m, ENU z)
        :param target_desc_rate: desired climb rate (m/s, ENU z, negative = descend), used
                                 when alt_pos_control is False
        :param marker_yaw:       AprilTag-measured pad yaw (rad, ENU), or None/NaN
                                 if no tag is currently detected
        :param cutoff:           bool — if True, command max descent and hold yaw (kill switch)
        :param alt_pos_control:  bool — if True, control altitude through a height P loop,
                                 false, command target_desc_rate directly
        :param quad_yaw:         drone yaw ψ (rad)
        :param quad_vel:         drone velocity v_a (m/s) — 3-vector [vx, vy, vz]
        :param u:                target-drone relative position p_m (ENU globally aligned)
                                 (m) — 3-vector [x, y, z]
        :param du:               target-drone velocity v_m (m/s) (ENU globally aligned)
                                — 3-vector [vx, vy, vz]
        :return: AttitudeTarget msg (attitude quaternion + thrust field interpreted by the
                 FCU as a normalised climb rate, 0.5 = hold)
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
                    alpha = (self.d_blend_start - dist_to_pad) / (
                        self.d_blend_start - self.d_blend_end
                    )
                    target_yaw = self._blend_angle(heading_to_pad, marker_yaw, alpha)
                else:
                    target_yaw = marker_yaw
            else:
                if dist_to_pad < self.d_hold_radius:
                    target_yaw = self.prev_yaw    # hold — don't chase a noisy/unavailable bearing near the pad
                else:
                    target_yaw = heading_to_pad   # still far out, LOS heading is fine

        # Condition yaw
        target_yaw = (target_yaw + np.pi) % (2 * np.pi) - np.pi  # type: ignore
        yaw_pre_slew = target_yaw
        target_yaw = self._slew_angle(target_yaw, self.prev_yaw, self.max_angle_rate, dt)
        slew_yaw = abs((target_yaw - yaw_pre_slew + np.pi) % (2 * np.pi) - np.pi) > 1e-9
        self.prev_yaw = target_yaw

        # Cutoff condition
        if cutoff:
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
            # Thrust field 0.0 = fastest descent the FCU allows (-WPNAV_SPEED_DN)
            msg.thrust = 0.0

            # Attitude was commanded level, so restart tilt slewing from zero on resume
            self.prev_phi = 0.0
            self.prev_theta = 0.0

            # Logging
            out = self._nan_outputs()
            out.update(
                F_x=0.0, F_y=0.0, F_z=0.0, F_thrust=0.0,
                climb_rate_cmd=-float(self.vz_dn_max), thrust_field=0.0,
                roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=float(target_yaw),
                slew_yaw=int(slew_yaw), marker_yaw_valid=int(tag_detected), cutoff_active=1,
            )
            self._last_outputs = out

            return msg

        # ---- PN/PD Controller (ENU) ----
        r = np.linalg.norm(u[:2])
        lam_gain_factor = 1 - np.exp(-self.drop_off_strength * r)

        gain_factor = self.terminal_gain * self.drop_off / (np.linalg.norm(u)**2 + self.drop_off)

        self.lam = self.lam_0 * lam_gain_factor
        self.Kp = self.Kp_0 * gain_factor
        self.Kd = self.Kd_0 * gain_factor

        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            accel_perp = np.zeros(3)
        else:
            omega = np.cross(u, du) / (u_norm**2)
            accel_perp = -self.lam * np.linalg.norm(du) * np.cross(u / u_norm, omega)

        accel_parallel = self.Kp * u + self.Kd * du
        accel = accel_perp + accel_parallel

        # ---- Altitude: outer loop only; FCU runs vel/accel/throttle ----
        z_err = target_altitude - (-u[QUAD_State.Z])    # height-above-pad error, +up

        if alt_pos_control:
            # Use descent and rate limits on position control (note target_descent rate is negative)
            lim = abs(target_desc_rate)
            vz_cmd = np.clip(self.Kp_pos_z * z_err, -lim, lim)
        else:
            vz_cmd = target_desc_rate # ENU: negative = descend

        if self.platform_vz_ff:
            # FCU climb rate is in the world frame, so add the pad's own vertical velocity
            # (du_z = pad_vz - drone_vz  ->  pad_vz = du_z + drone_vz)
            vz_cmd = vz_cmd + du[QUAD_State.Z] + quad_vel[QUAD_State.Z]

        vz_cmd_pre = vz_cmd
        vz_cmd = float(np.clip(vz_cmd, -self.vz_dn_max, self.vz_up_max))   # matches WPNAV_SPEED_DN/UP
        sat_climb_cmd = abs(vz_cmd - vz_cmd_pre) > 1e-9

        # Climb rate -> thrust field (0.5 = hold, 1.0 = full climb, 0.0 = full descent)
        if vz_cmd >= 0.0:
            thrust_field = 0.5 + 0.5 * vz_cmd / self.vz_up_max
        else:
            thrust_field = 0.5 + 0.5 * vz_cmd / self.vz_dn_max
        thrust_field = float(np.clip(thrust_field, 0.0, 1.0))

        # ---- Horizontal accel -> tilt ----
        drag_x = self.cD * quad_vel[QUAD_State.X] * abs(quad_vel[QUAD_State.X])
        drag_y = self.cD * quad_vel[QUAD_State.Y] * abs(quad_vel[QUAD_State.Y])

        F_x = self.m * accel[QUAD_State.X] + drag_x
        F_y = self.m * accel[QUAD_State.Y] + drag_y
        F_z = self.m * self.g  # FCU holds vertical force (it tilt-compensates throttle)

        thrust = np.sqrt(F_x**2 + F_y**2 + F_z**2)   # always >= m*g, so never ~0

        phi = np.arcsin(np.clip((F_x * np.sin(quad_yaw) - F_y * np.cos(quad_yaw)) / thrust, -1.0, 1.0))
        sat_roll = abs(phi) > self.max_tilt
        phi = float(np.clip(phi, -self.max_tilt, self.max_tilt))

        theta = np.arctan2(F_x * np.cos(quad_yaw) + F_y * np.sin(quad_yaw), F_z)
        sat_pitch = abs(theta) > self.max_tilt
        theta = float(np.clip(theta, -self.max_tilt, self.max_tilt))

        # Restrict/ramp outputs
        phi_pre, theta_pre = phi, theta
        phi = self._slew(phi, self.prev_phi, self.max_angle_rate, dt)
        theta = self._slew(theta, self.prev_theta, self.max_angle_rate, dt)
        self.prev_phi, self.prev_theta = phi, theta

        # ---- Stash outputs for logging ----
        out = self._nan_outputs()
        out.update(
            F_x=float(F_x), F_y=float(F_y), F_z=float(F_z), F_thrust=float(thrust),
            climb_rate_cmd=float(vz_cmd), thrust_field=float(thrust_field),
            roll_cmd=float(phi), pitch_cmd=float(theta), yaw_cmd=float(target_yaw),
            accel_perp_x=float(accel_perp[0]), accel_perp_y=float(accel_perp[1]),
            accel_par_x=float(accel_parallel[0]), accel_par_y=float(accel_parallel[1]),
            z_err=float(z_err),
            gain_factor=float(gain_factor), lam=float(self.lam),
            sat_climb_cmd=int(sat_climb_cmd),
            sat_roll_clamp=int(sat_roll), sat_pitch_clamp=int(sat_pitch),
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
        msg.thrust = thrust_field

        return msg


    def stop(self, stamp):
        """Make drone stop moving."""

        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        return msg


    def look(self, stamp):
        """Make drone yaw to search for target."""

        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.1

        return msg


    def update(
        self,
        dt,
        target_altitude,
        target_desc_rate,
        marker_yaw,
        cutoff,
        alt_pos_control,
        quad_yaw,
        quad_vel,
        landing_pad_relative_position,
        landing_pad_relative_velocity,
    ):
        """Update the controller with the current state and return a MAVROS command."""

        return self.controller(
            dt=dt,
            target_altitude=target_altitude,
            target_desc_rate=target_desc_rate,
            marker_yaw=marker_yaw,
            cutoff=cutoff,
            alt_pos_control=alt_pos_control,
            quad_yaw=quad_yaw,
            quad_vel=np.asarray(quad_vel),
            u=np.asarray(landing_pad_relative_position),
            du=np.asarray(landing_pad_relative_velocity),
        )


    def goPosition(self, stamp, lat, lon, alt, yaw=None):
        """Command the drone to fly to a global GPS position via MAVROS.

        lat, lon : degrees (WGS84)
        alt      : meters ABOVE HOME (FRAME_GLOBAL_REL_ALT)
        yaw      : optional heading in radians, ENU convention
                (0 = east, pi/2 = north). If None, yaw is ignored and
                the FCU holds its current heading.

        Must be called continuously (>2 Hz) while in GUIDED/OFFBOARD mode.
        """

        msg = GlobalPositionTarget()
        msg.header.stamp = stamp
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

        return msg


    def _slew(self, target, prev, max_rate, dt):
        """ Slew the PID output to the maximum rate.

        :param target: Target output
        :param prev: Previous output value
        :param max_rate: Maximum rate of change in output
        :return: Slewed output
        """
        max_step = max_rate * dt
        return prev + np.clip(target - prev, -max_step, max_step)


    def _slew_angle(self, target, prev, max_rate, dt):
        """ Slew the PID angle output to the maximum rate.

        :param target: Target output
        :param prev: Previous output value
        :param max_rate: Maximum rate of change in output
        :return: Slewed output
        """
        max_step = max_rate * dt
        diff = (target - prev + np.pi) % (2 * np.pi) - np.pi
        result = prev + np.clip(diff, -max_step, max_step)
        return (result + np.pi) % (2 * np.pi) - np.pi


    @staticmethod
    def _blend_angle(a, b, alpha):
        """Interpolate from angle a to b via unit vectors (shortest-arc, wrap-safe)."""
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