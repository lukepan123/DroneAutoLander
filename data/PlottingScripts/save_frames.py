#!/usr/bin/env python3
"""
save_frames.py  –  save frames from the car-mounted ghost camera to PNG
=======================================================================
Subscribes to the bridged Gazebo camera and writes frames to disk, named by their sim
time, for compositing into a ghosted landing sequence.

    python3 save_frames.py                         # every frame at up to 10 Hz -> ./frames
    python3 save_frames.py --rate 5 --out landing_55kmh
    python3 save_frames.py --start 40 --stop 60    # only sim time 40-60 s

Start it before the landing, Ctrl+C when done. Needs the bridge line
    "/ghost_cam/image@sensor_msgs/msg/Image[gz.msgs.Image"
in main_launch_sim.py, and use_sim_time (on by default here) so names match the logs.
"""

import argparse
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class FrameSaver(Node):
    def __init__(self, args):
        super().__init__('ghost_frame_saver',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.min_dt = 1.0 / args.rate if args.rate > 0 else 0.0
        self.start, self.stop = args.start, args.stop
        self.last_t = None
        self.n = 0
        self.bridge = CvBridge()
        self.create_subscription(Image, args.topic, self.callback, qos_profile_sensor_data)
        self.get_logger().info(f"Saving {args.topic} to {self.out.resolve()} "
                               f"at up to {args.rate:g} Hz")

    def callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.start is not None and t < self.start:
            return
        if self.stop is not None and t > self.stop:
            return
        if self.last_t is not None and t - self.last_t < self.min_dt:
            return
        self.last_t = t
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(str(self.out / f'frame_{t:010.3f}.png'), frame)
        self.n += 1
        if self.n % 20 == 0:
            self.get_logger().info(f"{self.n} frames saved (t = {t:.2f} s)")


def main():
    p = argparse.ArgumentParser(description='Save ghost-camera frames to PNG.')
    p.add_argument('--topic', default='/ghost_cam/image')
    p.add_argument('--out', default='frames', help='output folder (default: ./frames)')
    p.add_argument('--rate', type=float, default=10.0,
                   help='max frames saved per sim second (0 = every frame)')
    p.add_argument('--start', type=float, default=None, help='first sim time to save (s)')
    p.add_argument('--stop', type=float, default=None, help='last sim time to save (s)')
    args = p.parse_args()

    rclpy.init()
    node = FrameSaver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Done: {node.n} frames in {node.out.resolve()}")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
