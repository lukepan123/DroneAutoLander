#!/usr/bin/env python3
"""
AGV Controller with scripted test trajectories. [Claude]

Define a trajectory as a list of TrajectorySegment objects. Each segment
linearly ramps linear velocity (v) and angular velocity (w) between two
timestamps, e.g. "accelerate from v1 to v2 and turn from w1 to w2, between
t1 and t2 seconds". The controller samples the trajectory each control tick
and publishes it as a Twist, optionally with Gaussian noise added on top.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


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

    def sample(self, t: float):
        for seg in self.segments:
            if seg.contains(t):
                return seg.sample(t)

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


# Some test trajectories (for each case)
def no_trajectory() -> Trajectory:
    """ Stationary trajectory.
    """
    return Trajectory([
        TrajectorySegment(t_start=0.0,  t_end=1.0,  v_start=0.0,  v_end=0.0, w_start=0.0, w_end=0.0),
    ])

def straight_trajectory() -> Trajectory:
    """ Straight line trajectory.
    """
    return Trajectory([
        TrajectorySegment(t_start=0.0,  t_end=5.0,  v_start=0.0,  v_end=5.0, w_start=0.0, w_end=0.0),
    ])

def turn_trajectory() -> Trajectory:
    """ Turning trajectory.
    """
    return Trajectory([
        TrajectorySegment(t_start=0.0,  t_end=10.0,  v_start=0.0,  v_end=4.0, w_start=0.5, w_end=0.5),
    ])

def mix_trajectory1() -> Trajectory:
    """ Mixed trajectory.
    """
    return Trajectory([
        TrajectorySegment(t_start=0.0,  t_end=5.0,  v_start=0.0,  v_end=10.0, w_start=0.0, w_end=0.0),
        TrajectorySegment(t_start=5.0,  t_end=10.0, v_start=10.0, v_end=10.0, w_start=0.0, w_end=0.5),
        TrajectorySegment(t_start=10.0, t_end=20.0, v_start=10.0, v_end=4.0,  w_start=0.5, w_end=-0.5),
        TrajectorySegment(t_start=20.0, t_end=30.0, v_start=4.0,  v_end=10.0,  w_start=-0.5, w_end=0.0),
        TrajectorySegment(t_start=30.0, t_end=40.0, v_start=10.0,  v_end=20.0,  w_start=0.0, w_end=-0.2),
        TrajectorySegment(t_start=40.0, t_end=55.0, v_start=20.0,  v_end=5.0,  w_start=-0.2, w_end=0.0),
        TrajectorySegment(t_start=55.0, t_end=58.0, v_start=5.0,  v_end=5.0,  w_start=0.0, w_end=-0.5),
        TrajectorySegment(t_start=58.0, t_end=60.0, v_start=5.0,  v_end=5.0,  w_start=-0.5, w_end=-0.5),
        TrajectorySegment(t_start=60.0, t_end=65.0, v_start=5.0,  v_end=5.0,  w_start=-0.5, w_end=0.0),
        TrajectorySegment(t_start=65.0, t_end=75.0, v_start=5.0,  v_end=0.0,  w_start=0.0, w_end=0.0),
    ])

def mix_trajectory2() -> Trajectory:
    """ Mixed trajectory.
    """
    return Trajectory([
        TrajectorySegment(t_start=0.0,  t_end=10.0, v_start=0.0,  v_end=10.0, w_start=0.0, w_end=0.0),
        TrajectorySegment(t_start=10.0, t_end=15.0, v_start=10.0, v_end=10.0, w_start=0.0, w_end=-0.3),
        TrajectorySegment(t_start=15.0, t_end=45.0, v_start=10.0, v_end=10.0, w_start=-0.3, w_end=-0.3),
        TrajectorySegment(t_start=45.0, t_end=50.0, v_start=10.0, v_end=10.0, w_start=-0.3, w_end=0.0),
        TrajectorySegment(t_start=50.0, t_end=70.0, v_start=10.0, v_end=10.0, w_start=0.0, w_end=0.0),
        TrajectorySegment(t_start=70.0, t_end=75.0, v_start=10.0, v_end=10.0, w_start=0.0, w_end=0.3),
        TrajectorySegment(t_start=75.0, t_end=105.0, v_start=10.0, v_end=10.0, w_start=0.3, w_end=0.3),
        TrajectorySegment(t_start=105.0, t_end=110.0, v_start=10.0, v_end=10.0, w_start=0.3, w_end=0.0),
        TrajectorySegment(t_start=110.0, t_end=120.0, v_start=10.0, v_end=0.0, w_start=0.0, w_end=0.0),
    ])


class AGV_Controller(Node):
    def __init__(self,
                 trajectory: Optional[Trajectory] = None,
                 add_noise: bool = True,
                 linear_noise_std: float = 0.05,
                 angular_noise_std: float = 0.01,
                 angular_speed_limit: Optional[float] = 1.0,
                 linear_speed_limit: Optional[float] = None,
                 loop: bool = False):
        super().__init__('agv_controller')

        self.pub = self.create_publisher(Twist, '/cmd_rover_vel', 10)

        # --- trajectory + noise config ---
        self.trajectory = trajectory if trajectory is not None else no_trajectory()
        self.loop = loop

        self.add_noise = add_noise
        self.linear_noise_std = linear_noise_std
        self.angular_noise_std = angular_noise_std

        self.angular_speed_limit = angular_speed_limit
        self.linear_speed_limit = linear_speed_limit

        self.dt = 0.05  # 20 Hz
        self.elapsed = 0.0
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info(
            f"Trajectory duration: {self.trajectory.duration:.2f}s, "
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
        msg.linear.x = v
        msg.angular.z = w
        self.pub.publish(msg)

        self.get_logger().info(
            f"t={t:5.2f}s  v={v:6.2f} m/s  w={w:6.3f} rad/s",
            throttle_duration_sec=1.0
        )

        self.elapsed += self.dt


def main():
    rclpy.init()

    # Swap in your own list of TrajectorySegment(...) here to script a
    # different test run.
    trajectory = straight_trajectory()

    node = AGV_Controller(
        trajectory=trajectory,
        add_noise=False,
        linear_noise_std=0.05,
        angular_noise_std=0.01,
        angular_speed_limit=None,
        linear_speed_limit=None,
        loop=False,
    )

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()