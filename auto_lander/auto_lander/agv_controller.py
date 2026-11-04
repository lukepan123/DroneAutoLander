#!/usr/bin/env python3
"""
AGV Controller with scripted test trajectories.

Define a trajectory as a list of TrajectorySegment objects. Each segment
linearly ramps linear velocity (v) and angular velocity (w) between two
timestamps, e.g. "accelerate from v1 to v2 and turn from w1 to w2, between
t1 and t2 seconds". The controller samples the trajectory each control tick
and publishes it as a Twist, optionally with Gaussian noise added on top.

The trajectory is selected by ROS parameters so a batch runner can sweep it:
    trajectory            : none | straight | circle | sin | mbzirc | turn | mix1 | mix2
    speed_kmh             : cruise speed for straight / circle / sin (fixed scripts ignore it);
                            for mbzirc, 0 = competition speed schedule, > 0 = constant speed
    accel_distance_m      : mbzirc only: distance over which the car reaches its first speed.
                            The swept trajectories ramp at the Prius limits instead (below).
    turn_start_distance_m : straight cruise past the cruise line before any turn begins
    circle_yaw_rate       : rad/s, steady turn rate of the circle (same at every speed)
    sin_peak_yaw_rate     : rad/s, peak turn rate of the sine weave (same at every speed)
    sin_period            : s, period of the sine weave (same at every speed)
    sin_phase_deg         : deg, phase the weave starts at (run_batch.sh staggers it per repeat);
                            for mbzirc, the lap phase the car spawns at
    mbzirc_radius_m       : m, radius of the two MBZIRC figure-8 loops
    mbzirc_centre_distance_m : m, distance between the two loop centres
    mbzirc_duration_s     : s, run length before the car stops (competition slot: 20 min)
    add_noise             : add Gaussian noise to v and w

The mbzirc trajectory is the exception to the rule below: it is defined by the track
geometry (MBZIRC 2017 Challenge 1), so the yaw rate is v / R and changes with speed.
See mbzirc_trajectory() for the geometry, sources and matching SDF start poses.

Turns are defined by the steering command, not the path: the yaw-rate profile w(t) is
identical at every speed and only v changes. Heading dynamics seen by the filter are
therefore the same in every run, while lateral acceleration (v * w) and the radius /
weave amplitude scale linearly with speed.

Swept trajectories (straight / circle / sin) share one lead-in, laid out on the ground
relative to the UAV waiting point so every speed sees the same map:

    spawn(v) --- Prius ramp (d_ramp(v)) --- CRUISE LINE --- d_turn --- manoeuvre ...
                                            (fixed, CRUISE_LINE_BEFORE_UAV m before the UAV)

The ramp is jerk-limited at the Prius limits (PRIUS_*), so the car is at cruise speed
with zero acceleration by the cruise line, which sits well before the point where the
UAV first detects it. Only the spawn moves with speed: swept_world_poses(v) gives the
pose the SDF must use (run_batch.sh writes it per run, as for mbzirc).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np


KMH_TO_MS = 1.0 / 3.6


@dataclass
class TrajectorySegment:
    """One leg of a trajectory: linearly ramp v and w over [t_start, t_end]."""
    t_start: float
    t_end: float
    v_start: float
    v_end: float
    w_start: float = 0.0
    w_end: float = 0.0

    def contains(self, t: float) -> bool:
        return self.t_start <= t < self.t_end

    def sample(self, t: float):
        """Linearly interpolate (v, w) at time t within this segment."""
        span = max(self.t_end - self.t_start, 1e-9)
        alpha = float(np.clip((t - self.t_start) / span, 0.0, 1.0))
        v = self.v_start + alpha * (self.v_end - self.v_start)
        w = self.w_start + alpha * (self.w_end - self.w_start)
        return v, w


class Trajectory:
    """A time-ordered sequence of TrajectorySegments, sampled by elapsed time."""

    def __init__(self, segments: List[TrajectorySegment], hold_last: bool = True):
        self.segments = sorted(segments, key=lambda s: s.t_start)
        self.hold_last = hold_last
        self._starts = [s.t_start for s in self.segments]

    def sample(self, t: float):
        # Binary search (segments are contiguous), so long scripts such as the
        # 20-minute mbzirc run stay cheap at 20 Hz.
        i = bisect_right(self._starts, t) - 1
        if 0 <= i < len(self.segments) and self.segments[i].contains(t):
            return self.segments[i].sample(t)

        if not self.segments:
            return 0.0, 0.0

        if t < self.segments[0].t_start:
            return self.segments[0].v_start, self.segments[0].w_start

        # past the end of the last segment
        last = self.segments[-1]
        if self.hold_last:
            return last.v_end, last.w_end
        return 0.0, 0.0

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        return max(s.t_end for s in self.segments)


# ---------------------------------------------------------------------------
# Vehicle limits: Toyota Prius (XW30, the generation the prius_hybrid model depicts).
# Keep the AckermannSteering plugin in prius_hybrid/model.sdf equal to these.
# ---------------------------------------------------------------------------
PRIUS_MAX_ACCEL = 2.7    # m/s^2, mean 0-100 km/h (10.4 s, cars-data.com, XW30 1.8 HSD)
PRIUS_MAX_DECEL = 9.5    # m/s^2, km77 measured 60-0 in 14.5 m / 120-0 in 56.4 m (XW50)
PRIUS_MAX_JERK = 2.0     # m/s^3, assumed (typical comfortable passenger-car jerk); no spec
PRIUS_MAX_SPEED = 50.0   # m/s, 180 km/h top speed
# Commanded ramp stays this fraction inside the plugin limits, so the plugin's speed
# limiter never clips the command and the car follows the commanded timeline exactly.
CMD_LIMIT_MARGIN = 0.95

# ---------------------------------------------------------------------------
# Swept-trajectory layout (v in m/s), in the iris_runway_new world.
# ---------------------------------------------------------------------------
UAV_WAIT_WORLD = (-100.0, -290.0)   # m, UAV spawn = default GPS waiting point
CAR_HEADING_WORLD_DEG = 90.0        # car drives world +y, straight under the UAV
# Cruise line: the car is at cruise speed (zero acceleration) from here on. The UAV first
# detected the car 17.6-22.9 m before its waiting point in the 60 / 65 km/h logs, so
# 35 m leaves >= 12 m (>= 0.6 s at 70 km/h) of constant-speed driving before detection.
CRUISE_LINE_BEFORE_UAV = 35.0       # m
STATIONARY_BEFORE_UAV = 30.0        # m, 0 km/h runs keep the original spawn (-100, -320)
# Straight cruise past the cruise line before a turn: 35 m puts the turn-in abeam the
# UAV waiting point, where the old 10 m + 20 m lead-in intended it to be.
TURN_START_DISTANCE = 20.0          # m
ACCEL_DISTANCE = 10.0               # m, mbzirc only (its own ramp from rest)
CIRCLE_YAW_RATE = 0.2               # rad/s
SIN_PEAK_YAW_RATE = 0.2             # rad/s, matches the circle so peak lateral accel matches too
SIN_PERIOD = 12.0                   # s
SIN_PHASE_DEG = 0.0                 # deg
MIN_RAMP_TIME = 0.05                # s, one control tick (mbzirc accel_distance = 0 -> near-instant)
RAMP_DT = 0.05                      # s, ramp sampled once per control tick


def ramp_profile(v: float,
                 a_max: float = CMD_LIMIT_MARGIN * PRIUS_MAX_ACCEL,
                 j_max: float = CMD_LIMIT_MARGIN * PRIUS_MAX_JERK):
    """Jerk-limited (S-curve) ramp from rest to v.

    Acceleration rises at j_max, holds a_max, then falls at j_max to exactly zero as the
    car reaches v, so there is no acceleration step at the end of the ramp. If v is too
    low to reach a_max (v < a_max^2 / j_max), the peak is sqrt(v * j_max) instead.

    :return: (a_peak, t_jerk, t_const, t_ramp, d_ramp). The profile is symmetric, so
             d_ramp = v * t_ramp / 2.
    """
    a = min(a_max, float(np.sqrt(v * j_max)))
    t_jerk = a / j_max
    t_const = max(v / a - t_jerk, 0.0)
    t_ramp = 2.0 * t_jerk + t_const
    return a, t_jerk, t_const, t_ramp, 0.5 * v * t_ramp


def _ramp_speed(t: np.ndarray, v: float, a: float, t_jerk: float, t_const: float,
                j_max: float) -> np.ndarray:
    """Speed along the S-curve from ramp_profile() at times t (s, from ramp start)."""
    t = np.asarray(t, dtype=float)
    t_ramp = 2.0 * t_jerk + t_const
    rise = 0.5 * j_max * t**2                                   # jerk-up
    hold = 0.5 * a * t_jerk + a * (t - t_jerk)                  # constant accel
    fall = v - 0.5 * j_max * np.clip(t_ramp - t, 0.0, None)**2  # jerk-down
    return np.select([t < t_jerk, t < t_jerk + t_const, t < t_ramp], [rise, hold, fall], v)


def _lead_in(v: float, d_turn: float = 0.0):
    """Prius S-curve ramp 0 -> v, then cruise d_turn metres past the cruise line.

    Returns the segments and the time the manoeuvre may start.
    """
    j = CMD_LIMIT_MARGIN * PRIUS_MAX_JERK
    a, t_jerk, t_const, t_ramp, _ = ramp_profile(v, CMD_LIMIT_MARGIN * PRIUS_MAX_ACCEL, j)
    n = max(int(np.ceil(t_ramp / RAMP_DT)), 1)
    t = np.linspace(0.0, t_ramp, n + 1)
    vs = _ramp_speed(t, v, a, t_jerk, t_const, j)
    vs[-1] = v
    segments = [TrajectorySegment(t_start=float(t0), t_end=float(t1),
                                  v_start=float(v0), v_end=float(v1))
                for t0, t1, v0, v1 in zip(t[:-1], t[1:], vs[:-1], vs[1:])]
    t_turn = t_ramp + max(d_turn, 0.0) / v
    if t_turn > t_ramp:
        segments.append(TrajectorySegment(t_start=t_ramp, t_end=t_turn, v_start=v, v_end=v))
    return segments, t_turn


def swept_world_poses(v: float,
                      uav=UAV_WAIT_WORLD,
                      heading_deg: float = CAR_HEADING_WORLD_DEG,
                      cruise_line_before_uav: float = CRUISE_LINE_BEFORE_UAV):
    """Gazebo world spawn (x, y, yaw[deg]) of the car for a swept run at v (m/s).

    The car spawns d_ramp(v) behind the cruise line, so it reaches cruise exactly there
    whatever v is. v <= 0 (stationary runs) keeps the original spawn, so the existing
    0 km/h results stay valid.
    """
    h = np.radians(heading_deg)
    if v > 0.0:
        back = cruise_line_before_uav + ramp_profile(v)[4]
    else:
        back = STATIONARY_BEFORE_UAV
    return (float(uav[0] - back * np.cos(h)), float(uav[1] - back * np.sin(h)), float(heading_deg))


def no_trajectory() -> Trajectory:
    """Stationary trajectory."""
    return Trajectory([
        TrajectorySegment(t_start=0.0, t_end=1.0, v_start=0.0, v_end=0.0),
    ])


def straight_trajectory(v: float,
                        d_turn: float = TURN_START_DISTANCE,
                        **_) -> Trajectory:
    """Prius ramp to v, then hold it in a straight line (d_turn is moot)."""
    segments, _ = _lead_in(v)
    return Trajectory(segments)


def circle_trajectory(v: float,
                      d_turn: float = TURN_START_DISTANCE,
                      circle_yaw_rate: float = CIRCLE_YAW_RATE,
                      t_turn_in: float = 3.0,
                      **_) -> Trajectory:
    """Prius ramp to v, cruise d_turn past the cruise line, ease into a constant
    yaw-rate left turn over t_turn_in, then hold it.

    The yaw rate is the same at every speed, so radius = v / w and lateral
    acceleration = v * w both grow linearly with v (0.2 rad/s: 7 m and 0.3 m/s^2 at
    5 km/h, 97 m and 3.9 m/s^2 at 70 km/h).
    """
    segments, t1 = _lead_in(v, d_turn)
    segments.append(TrajectorySegment(t_start=t1, t_end=t1 + t_turn_in, v_start=v, v_end=v,
                                      w_start=0.0, w_end=circle_yaw_rate))
    return Trajectory(segments)


def sin_trajectory(v: float,
                   d_turn: float = TURN_START_DISTANCE,
                   sin_peak_yaw_rate: float = SIN_PEAK_YAW_RATE,
                   sin_period: float = SIN_PERIOD,
                   sin_phase_deg: float = SIN_PHASE_DEG,
                   duration: float = 300.0,
                   dt: float = 0.25,
                   **_) -> Trajectory:
    """Prius ramp to v, cruise d_turn past the cruise line, then a sinusoidal weave at
    constant v.

    Heading follows theta(t) = theta_max * sin(omega * t + phase), eased in over the
    first half-period so w(0) = 0 whatever the phase, with theta_max =
    sin_peak_yaw_rate / omega. The phase only moves where in the cycle the weave
    starts; the shape, amplitude and yaw-rate peak are unchanged. The yaw-rate
    profile is therefore the same at every speed; the lateral amplitude
    (~ v * theta_max / omega) and peak lateral acceleration (v * peak yaw rate) scale
    linearly with v. The weave runs for `duration` s so it never degenerates into a
    circle mid-run.
    """
    omega = 2.0 * np.pi / sin_period
    theta_max = sin_peak_yaw_rate / omega
    phi = np.radians(sin_phase_deg)

    # theta(t) = theta_max * env(t) * sin(omega t), env = smoothstep over t_ramp
    t = np.linspace(0.0, duration, int(np.ceil(duration / dt)) + 1)
    t_ramp = 0.5 * sin_period
    s = np.clip(t / t_ramp, 0.0, 1.0)
    env = s * s * (3.0 - 2.0 * s)
    d_env = 6.0 * s * (1.0 - s) / t_ramp
    w = theta_max * (d_env * np.sin(omega * t + phi) + env * omega * np.cos(omega * t + phi))

    # Prius lead-in, then the sine shifted to start after it
    segments, t_straight = _lead_in(v, d_turn)
    t += t_straight
    segments += [
        TrajectorySegment(t_start=float(t0), t_end=float(t1), v_start=v, v_end=v,
                          w_start=float(w0), w_end=float(w1))
        for t0, t1, w0, w1 in zip(t[:-1], t[1:], w[:-1], w[1:])
    ]
    return Trajectory(segments)


# ---------------------------------------------------------------------------
# MBZIRC 2017 Challenge 1: figure-8 track
# ---------------------------------------------------------------------------
# Track: two circular loops of radius R, centres D apart on the arena x axis, joined
# by two straights that cross at the arena centre and meet each loop tangentially.
# Arena frame: origin at arena centre, x along the 90 m side, y along the 60 m side.
#
# Defaults R = 16 m, D = 54 m are reconstructed, not copied from the rulebook (the
# official document is no longer online). They satisfy what teams reported from the
# final event: an 86 x 32 m track inside the 90 x 60 m arena, ~27 s per half lap at
# 15 km/h (NimbRo, JFR 2019) and ~1 m/s^2 in the curves (NimbRo, ECMR 2017).
# JSK's simulator used the 2016 provisional geometry R = 20 m, D = 55 m (95 m long,
# 32 s per half lap); set the two parameters to those values to reproduce it.
#
# Speed: 15 km/h, slowed after 8 minutes (NimbRo). The 10 -> 5 km/h steps follow the
# provisional rules; the 16-minute timing of the second step is an assumption.
MBZIRC_RADIUS = 16.0            # m
MBZIRC_CENTRE_DISTANCE = 54.0   # m
MBZIRC_DURATION = 20.0 * 60.0   # s, one competition slot; the car stops afterwards
MBZIRC_SPEED_SCHEDULE = ((0.0, 15.0), (480.0, 10.0), (960.0, 5.0))  # (t [s], km/h)
MBZIRC_SPEED_RAMP = 3.0         # s, linear ramp between scheduled speeds
# Where the arena sits in the Gazebo world (iris_mbzirc.sdf): centre and the world
# heading of the arena x axis. run_batch.sh writes per-run worlds from these.
MBZIRC_ARENA_CENTRE_WORLD = (-100.0, -260.0)  # m
MBZIRC_ARENA_YAW_WORLD_DEG = 90.0             # arena x axis points world +y (north)


def mbzirc_geometry(radius: float = MBZIRC_RADIUS,
                    centre_distance: float = MBZIRC_CENTRE_DISTANCE):
    """Return (theta, l_straight, l_arc, lap_length) for the figure-8.

    theta is the angle between the arena x axis and the line from the centre to a
    tangent point (cos theta = 2R / D). Each straight is D sin(theta) long, each loop
    sweeps 2(pi - theta) of arc.
    """
    if 2.0 * radius >= centre_distance:
        raise ValueError("MBZIRC track needs centre_distance > 2 * radius")
    theta = float(np.arccos(2.0 * radius / centre_distance))
    l_straight = centre_distance * np.sin(theta)
    l_arc = 2.0 * radius * (np.pi - theta)
    return theta, l_straight, l_arc, 2.0 * (l_straight + l_arc)


def mbzirc_track_pose(s: float, radius: float = MBZIRC_RADIUS,
                      centre_distance: float = MBZIRC_CENTRE_DISTANCE):
    """Arena-frame pose (x, y, yaw[rad]) at arc length s along the lap.

    s = 0 is the left-loop tangent point, facing down the straight through the arena
    centre (+x, -y). The lap then runs the right loop anticlockwise, the second
    straight, and the left loop clockwise.
    """
    theta, l_straight, l_arc, lap = mbzirc_geometry(radius, centre_distance)
    h = 0.5 * l_straight
    x, y, yaw = -h * np.sin(theta), h * np.cos(theta), theta - 0.5 * np.pi
    s = float(np.mod(s, lap))
    for length, curv in ((l_straight, 0.0), (l_arc, 1.0 / radius),
                         (l_straight, 0.0), (l_arc, -1.0 / radius)):
        u = min(s, length)
        if curv == 0.0:
            x, y = x + u * np.cos(yaw), y + u * np.sin(yaw)
        else:  # exact circular arc
            x += (np.sin(yaw + curv * u) - np.sin(yaw)) / curv
            y -= (np.cos(yaw + curv * u) - np.cos(yaw)) / curv
            yaw += curv * u
        s -= u
        if s <= 0.0:
            break
    return float(x), float(y), float(np.arctan2(np.sin(yaw), np.cos(yaw)))


def mbzirc_start_poses(radius: float = MBZIRC_RADIUS,
                       centre_distance: float = MBZIRC_CENTRE_DISTANCE,
                       phase_deg: float = 0.0):
    """Arena-frame start poses (x, y, yaw[rad]) the SDF must match.

    Car: at rest on the track, phase_deg of a lap past the left-loop tangent point
    (0 deg). Staggering the phase across repeats samples where the car is when the
    UAV arrives, as the competition did (the car was already lapping at the start
    signal). The UAV start never moves.
    UAV: centre of the left loop, where NimbRo reports the MAV started in 2017.
    """
    lap = mbzirc_geometry(radius, centre_distance)[3]
    car = mbzirc_track_pose(phase_deg / 360.0 * lap, radius, centre_distance)
    uav = (-0.5 * centre_distance, 0.0, 0.0)
    return car, uav


def mbzirc_world_poses(phase_deg: float = 0.0,
                       radius: float = MBZIRC_RADIUS,
                       centre_distance: float = MBZIRC_CENTRE_DISTANCE,
                       centre=MBZIRC_ARENA_CENTRE_WORLD,
                       arena_yaw_deg: float = MBZIRC_ARENA_YAW_WORLD_DEG):
    """Gazebo world poses (x, y, yaw[deg]) for the car and UAV, for writing the SDF."""
    a = np.radians(arena_yaw_deg)
    c, s_ = np.cos(a), np.sin(a)

    def to_world(p):
        x, y, yaw = p
        yaw_w = (np.degrees(yaw) + arena_yaw_deg + 180.0) % 360.0 - 180.0
        return (centre[0] + c * x - s_ * y, centre[1] + s_ * x + c * y, float(yaw_w))

    car, uav = mbzirc_start_poses(radius, centre_distance, phase_deg)
    return to_world(car), to_world(uav)


def _mbzirc_heading_change(s: np.ndarray, radius: float, l_straight: float,
                           l_arc: float, lap: float) -> np.ndarray:
    """Integral of path curvature from the start to arc length s (zero net per lap)."""
    sweep = l_arc / radius
    s = np.mod(s, lap)
    b1, b2, b3 = l_straight, l_straight + l_arc, 2.0 * l_straight + l_arc
    return np.select(
        [s < b1, s < b2, s < b3],
        [0.0, (s - b1) / radius, sweep],
        default=sweep - (s - b3) / radius,  # second loop turns the other way
    )


def _speed_profile(t: np.ndarray, schedule: Sequence[Tuple[float, float]],
                   t_accel: float, ramp: float) -> np.ndarray:
    """Speed [m/s] vs time: linear rise from rest over t_accel, then the schedule
    with linear ramps between levels."""
    v = np.full_like(t, schedule[0][1] * KMH_TO_MS)
    for (_, v_prev), (t_k, v_k) in zip(schedule[:-1], schedule[1:]):
        a, b = v_prev * KMH_TO_MS, v_k * KMH_TO_MS
        alpha = np.clip((t - t_k) / max(ramp, 1e-9), 0.0, 1.0)
        v = np.where(t >= t_k, a + alpha * (b - a), v)
    return np.where(t < t_accel, v * t / t_accel, v)


def mbzirc_trajectory(v: float = 0.0,
                      d_accel: float = ACCEL_DISTANCE,
                      mbzirc_radius_m: float = MBZIRC_RADIUS,
                      mbzirc_centre_distance_m: float = MBZIRC_CENTRE_DISTANCE,
                      mbzirc_duration_s: float = MBZIRC_DURATION,
                      sin_phase_deg: float = 0.0,
                      dt: float = 0.05,
                      **_) -> Trajectory:
    """Laps of the MBZIRC 2017 figure-8, open loop, from mbzirc_start_poses().

    v <= 0 follows MBZIRC_SPEED_SCHEDULE; v > 0 holds that speed for the whole run
    (for speed sweeps). The car accelerates from rest over d_accel metres along the
    track, so the path is unchanged by the lead-in.

    sin_phase_deg is reused as the lap phase the car starts at (the batch already
    staggers it per repeat), so no launch-file change is needed. The car must be
    spawned at mbzirc_start_poses(..., phase_deg) for the path to line up;
    run_batch.sh does this by writing a per-run world.

    Segments are one control tick long with constant (v, w) chosen so that v*dt and
    w*dt are the exact distance and heading change over the tick: an ideal unicycle
    tracks the track with no integration drift. The real car still drifts (open
    loop), so check the path closes after a few laps.
    """
    radius, dist = mbzirc_radius_m, mbzirc_centre_distance_m
    _, l_straight, l_arc, lap = mbzirc_geometry(radius, dist)
    schedule = MBZIRC_SPEED_SCHEDULE if v <= 0.0 else ((0.0, v / KMH_TO_MS),)
    v_cruise = schedule[0][1] * KMH_TO_MS
    s0 = sin_phase_deg / 360.0 * lap  # arc length of the spawn point
    t_accel = max(2.0 * d_accel / v_cruise, MIN_RAMP_TIME)

    # Distance along the track: integrate v(t) on a 10x finer grid (trapezoid)
    sub = 10
    n = int(np.ceil(mbzirc_duration_s / dt))
    t_fine = np.linspace(0.0, n * dt, n * sub + 1)
    v_fine = _speed_profile(t_fine, schedule, t_accel, MBZIRC_SPEED_RAMP)
    s_fine = np.concatenate(([0.0], np.cumsum(0.5 * (v_fine[1:] + v_fine[:-1]) * np.diff(t_fine))))
    t, s = t_fine[::sub], s_fine[::sub]

    # Unwrapped heading change: per-lap offset is zero, so the modular form is continuous
    psi = _mbzirc_heading_change(s0 + s, radius, l_straight, l_arc, lap)
    v_seg = np.diff(s) / dt
    w_seg = np.diff(psi) / dt

    # Each segment is centred on its control tick (shifted back half a tick). The node's
    # clock is a running sum of dt, so a tick can land a hair before t_k; with
    # tick-aligned boundaries that repeats one segment and skips the next, and a skipped
    # straight-to-arc segment leaves a permanent heading error (~0.7 m off-track).
    half = 0.5 * dt
    return Trajectory([
        TrajectorySegment(t_start=float(t0) - half, t_end=float(t1) - half,
                          v_start=float(vs), v_end=float(vs), w_start=float(ws), w_end=float(ws))
        for t0, t1, vs, ws in zip(t[:-1], t[1:], v_seg, w_seg)
    ], hold_last=False)  # stop when the slot ends instead of circling forever


# ---------------------------------------------------------------------------
# Fixed scripts (speed_kmh is ignored)
# ---------------------------------------------------------------------------
def turn_trajectory() -> Trajectory:
    """Turning trajectory."""
    return Trajectory([
        TrajectorySegment(t_start=0.0, t_end=4.0, v_start=0.0, v_end=15.0, w_start=0.0, w_end=0.0),
        TrajectorySegment(t_start=4.0, t_end=4.5, v_start=0.0, v_end=15.0, w_start=0.0, w_end=-0.2),
    ])


def mix_trajectory1() -> Trajectory:
    """Mixed trajectory."""
    return Trajectory([
        TrajectorySegment(t_start=0.0,  t_end=5.0,  v_start=0.0,  v_end=10.0, w_start=0.0,  w_end=0.0),
        TrajectorySegment(t_start=5.0,  t_end=10.0, v_start=10.0, v_end=10.0, w_start=0.0,  w_end=0.5),
        TrajectorySegment(t_start=10.0, t_end=20.0, v_start=10.0, v_end=4.0,  w_start=0.5,  w_end=-0.5),
        TrajectorySegment(t_start=20.0, t_end=30.0, v_start=4.0,  v_end=10.0, w_start=-0.5, w_end=0.0),
        TrajectorySegment(t_start=30.0, t_end=40.0, v_start=10.0, v_end=20.0, w_start=0.0,  w_end=-0.2),
        TrajectorySegment(t_start=40.0, t_end=60.0, v_start=20.0, v_end=5.0,  w_start=-0.2, w_end=0.0),
        TrajectorySegment(t_start=60.0, t_end=63.0, v_start=5.0,  v_end=5.0,  w_start=0.0,  w_end=-0.5),
        TrajectorySegment(t_start=63.0, t_end=65.0, v_start=5.0,  v_end=5.0,  w_start=-0.5, w_end=-0.5),
        TrajectorySegment(t_start=65.0, t_end=70.0, v_start=5.0,  v_end=5.0,  w_start=-0.5, w_end=0.0),
        TrajectorySegment(t_start=70.0, t_end=80.0, v_start=5.0,  v_end=0.0,  w_start=0.0,  w_end=0.0),
    ])


def mix_trajectory2() -> Trajectory:
    """Mixed trajectory."""
    return Trajectory([
        TrajectorySegment(t_start=0.0,   t_end=10.0,  v_start=0.0,  v_end=10.0, w_start=0.0,  w_end=0.0),
        TrajectorySegment(t_start=10.0,  t_end=15.0,  v_start=10.0, v_end=10.0, w_start=0.0,  w_end=-0.3),
        TrajectorySegment(t_start=15.0,  t_end=45.0,  v_start=10.0, v_end=10.0, w_start=-0.3, w_end=-0.3),
        TrajectorySegment(t_start=45.0,  t_end=50.0,  v_start=10.0, v_end=10.0, w_start=-0.3, w_end=0.0),
        TrajectorySegment(t_start=50.0,  t_end=70.0,  v_start=10.0, v_end=10.0, w_start=0.0,  w_end=0.0),
        TrajectorySegment(t_start=70.0,  t_end=75.0,  v_start=10.0, v_end=10.0, w_start=0.0,  w_end=0.3),
        TrajectorySegment(t_start=75.0,  t_end=105.0, v_start=10.0, v_end=10.0, w_start=0.3,  w_end=0.3),
        TrajectorySegment(t_start=105.0, t_end=110.0, v_start=10.0, v_end=10.0, w_start=0.3,  w_end=0.0),
        TrajectorySegment(t_start=110.0, t_end=120.0, v_start=10.0, v_end=0.0,  w_start=0.0,  w_end=0.0),
    ])


SWEPT_TRAJECTORIES = {
    "straight": straight_trajectory,
    "circle": circle_trajectory,
    "sin": sin_trajectory,
}
FIXED_TRAJECTORIES = {
    "none": no_trajectory,
    "turn": turn_trajectory,
    "mix1": mix_trajectory1,
    "mix2": mix_trajectory2,
}


def make_trajectory(name: str, v: float,
                    d_accel: float = ACCEL_DISTANCE,
                    d_turn: float = TURN_START_DISTANCE,
                    **shape) -> Trajectory:
    """Build a trajectory by name; v (m/s) and d_turn only apply to the swept ones,
    d_accel only to mbzirc (the fixed scripts carry their own timing)."""
    name = name.strip().lower()
    if name == "mbzirc":
        # v <= 0 is valid here: it selects the competition speed schedule
        return mbzirc_trajectory(v, d_accel=d_accel, **shape)
    if name in SWEPT_TRAJECTORIES:
        if v <= 0.0:
            return no_trajectory()
        if v > CMD_LIMIT_MARGIN * PRIUS_MAX_SPEED:
            raise ValueError(f"{v:.1f} m/s is above the Prius speed limit used in the SDF")
        return SWEPT_TRAJECTORIES[name](v, d_turn=d_turn, **shape)
    if name in FIXED_TRAJECTORIES:
        return FIXED_TRAJECTORIES[name]()
    valid = sorted(SWEPT_TRAJECTORIES) + ["mbzirc"] + sorted(FIXED_TRAJECTORIES)
    raise ValueError(f"Unknown trajectory '{name}'. Valid: {valid}")


class AGV_Controller(Node):
    def __init__(self,
                 trajectory: Optional[Trajectory] = None,
                 add_noise: Optional[bool] = None,
                 linear_noise_std: float = 0.05,
                 angular_noise_std: float = 0.01,
                 angular_speed_limit: Optional[float] = 1.0,
                 linear_speed_limit: Optional[float] = None,
                 loop: bool = False):
        super().__init__('agv_controller')

        # --- parameters (used when no trajectory / noise flag is passed in) ---
        self.declare_parameter('trajectory', 'none')
        self.declare_parameter('speed_kmh', 0.0)
        self.declare_parameter('accel_distance_m', ACCEL_DISTANCE)
        self.declare_parameter('turn_start_distance_m', TURN_START_DISTANCE)
        self.declare_parameter('circle_yaw_rate', CIRCLE_YAW_RATE)
        self.declare_parameter('sin_peak_yaw_rate', SIN_PEAK_YAW_RATE)
        self.declare_parameter('sin_period', SIN_PERIOD)
        self.declare_parameter('sin_phase_deg', SIN_PHASE_DEG)
        self.declare_parameter('mbzirc_radius_m', MBZIRC_RADIUS)
        self.declare_parameter('mbzirc_centre_distance_m', MBZIRC_CENTRE_DISTANCE)
        self.declare_parameter('mbzirc_duration_s', MBZIRC_DURATION)
        self.declare_parameter('add_noise', False)
        traj_name = self.get_parameter('trajectory').get_parameter_value().string_value
        speed_kmh = self.get_parameter('speed_kmh').get_parameter_value().double_value
        d_accel = self.get_parameter('accel_distance_m').get_parameter_value().double_value
        d_turn = self.get_parameter('turn_start_distance_m').get_parameter_value().double_value

        self.pub = self.create_publisher(Twist, '/cmd_rover_vel', 10)

        # --- trajectory + noise config ---
        if trajectory is None:
            shape = {k: self.get_parameter(k).get_parameter_value().double_value
                     for k in ('circle_yaw_rate', 'sin_peak_yaw_rate', 'sin_period', 'sin_phase_deg',
                               'mbzirc_radius_m', 'mbzirc_centre_distance_m', 'mbzirc_duration_s')}
            trajectory = make_trajectory(traj_name, speed_kmh * KMH_TO_MS, d_accel, d_turn, **shape)
            self.get_logger().info(f"Turn shape: {shape}")
            name = traj_name.strip().lower()
            if name == 'mbzirc':
                R, D = shape['mbzirc_radius_m'], shape['mbzirc_centre_distance_m']
                phase = shape['sin_phase_deg']
                car, uav = mbzirc_world_poses(phase, R, D)
                lap = mbzirc_geometry(R, D)[3]
                self.get_logger().info(
                    f"MBZIRC lap {lap:.1f} m, start phase {phase:.1f} deg, "
                    f"{'competition schedule' if speed_kmh <= 0 else f'{speed_kmh:.1f} km/h'}. "
                    f"Expected world spawn (x, y, yaw): car ({car[0]:.3f}, {car[1]:.3f}, "
                    f"{car[2]:.3f} deg), UAV ({uav[0]:.3f}, {uav[1]:.3f}). "
                    f"The SDF poses must match these.")
            elif name in SWEPT_TRAJECTORIES:
                v = speed_kmh * KMH_TO_MS
                car = swept_world_poses(v)
                if v > 0.0:
                    a, _, _, t_ramp, d_ramp = ramp_profile(v)
                    ramp = (f"Prius ramp {t_ramp:.2f} s / {d_ramp:.1f} m "
                            f"(peak {a:.2f} m/s^2), cruise from {CRUISE_LINE_BEFORE_UAV:.0f} m "
                            f"before the UAV, turn-in {d_turn:.0f} m later. ")
                else:
                    ramp = "Stationary. "
                self.get_logger().info(
                    f"{ramp}Expected world spawn (x, y, yaw): car ({car[0]:.3f}, "
                    f"{car[1]:.3f}, {car[2]:.3f} deg), UAV ({UAV_WAIT_WORLD[0]:.3f}, "
                    f"{UAV_WAIT_WORLD[1]:.3f}). The SDF poses must match these.")
        self.trajectory = trajectory
        self.loop = loop

        if add_noise is None:
            add_noise = self.get_parameter('add_noise').get_parameter_value().bool_value
        self.add_noise = add_noise
        self.linear_noise_std = linear_noise_std
        self.angular_noise_std = angular_noise_std

        self.angular_speed_limit = angular_speed_limit
        self.linear_speed_limit = linear_speed_limit

        self.dt = 0.05  # 20 Hz
        self.elapsed = 0.0
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info(
            f"Trajectory '{traj_name}' at {speed_kmh:.1f} km/h "
            f"(turns after +{d_turn:.0f} m past the cruise line), "
            f"duration: {self.trajectory.duration:.2f}s, "
            f"loop={self.loop}, noise={self.add_noise}"
        )

    def update(self):
        t = self.elapsed
        if self.loop and self.trajectory.duration > 0:
            t = t % self.trajectory.duration

        v, w = self.trajectory.sample(t)

        if self.add_noise:
            v += np.random.normal(0.0, self.linear_noise_std)
            w += np.random.normal(0.0, self.angular_noise_std)

        if self.linear_speed_limit is not None:
            v = float(np.clip(v, 0.0, self.linear_speed_limit))
        if self.angular_speed_limit is not None:
            w = float(np.clip(w, -self.angular_speed_limit, self.angular_speed_limit))

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

        self.get_logger().info(
            f"t={t:5.2f}s  v={v:6.2f} m/s  w={w:6.3f} rad/s",
            throttle_duration_sec=1.0
        )

        self.elapsed += self.dt


def main():
    rclpy.init()

    # trajectory=None -> chosen by the 'trajectory' / 'speed_kmh' parameters.
    # Pass a Trajectory([...]) here instead to hard-code a one-off test.
    node = AGV_Controller(
        trajectory=None,
        add_noise=None,
        linear_noise_std=0.05,
        angular_noise_std=0.01,
        angular_speed_limit=None,
        linear_speed_limit=None,
        loop=False,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()