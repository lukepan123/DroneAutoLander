#!/usr/bin/env python3
"""Gimbal camera (RTSP) -> /camera/image_raw for the auto_lander perception nodes.

The lander's YOLO node subscribes to /camera/image_raw and nothing in that repo
publishes it for an RTSP camera (README, "No camera publisher"). This reads the
SIYI stream with OpenCV/FFmpeg, keeps only the newest frame, scales it to
width x height (16:9, the size the YOLO node processes) and publishes it bgr8,
best-effort, stamped latency_s before the frame was read (the camera's own delay).

    python3 rtsp_camera.py --ros-args -p url:=rtsp://192.168.144.25:8554/main.264 \
        -p width:=640 -p height:=360 -p max_fps:=15.0 -p fisheye_f:=215.0 -p out_hfov_deg:=120.0 \
        -p latency_s:=0.2

Fisheye -> pinhole. The A2 mini's picture is a fisheye (about 230 px per radian at
640 px wide, measured in flight on 2026-11-13 from the car's apparent size against
its RTK range), and the lander's vision assumes an ideal pinhole. Fed the raw
picture with a "160 degree pinhole" model, every range came out about four times
too short. With fisheye_f > 0 each frame is remapped to a true pinhole image of
out_hfov_deg, so the launch file's CAMERA_HFOV_RAD must equal out_hfov_deg and its
distortion stays zero.
    fisheye_f      px per radian of the source picture, scaled to a 640 px wide
                   frame (ideal equidistant lens: radius = f * angle). 0 = pass the
                   frames through unchanged. A value BELOW the true one makes
                   objects look bigger, i.e. closer than they are (the safe side).
"""
import os
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0")
import cv2                                    # noqa: E402
import numpy as np                            # noqa: E402
import rclpy                                  # noqa: E402
from rclpy.duration import Duration           # noqa: E402
from rclpy.node import Node                   # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy   # noqa: E402
from sensor_msgs.msg import Image             # noqa: E402


def build_rectify_maps(src_w, src_h, fisheye_f, out_w, out_h, out_hfov_deg):
    """Remap tables (for cv2.remap) from an equidistant fisheye source to a pinhole
    picture. fisheye_f is in px/rad for a 640 px wide source; optical centre = image
    centre in both."""
    f_src = float(fisheye_f) * src_w / 640.0
    f_out = (out_w / 2.0) / np.tan(np.radians(out_hfov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x = (u - (out_w - 1) / 2.0) / f_out
    y = (v - (out_h - 1) / 2.0) / f_out
    r = np.sqrt(x * x + y * y)
    scale = np.where(r > 1e-6, np.arctan(r) / np.maximum(r, 1e-6), 1.0)   # angle / tan(angle)
    map_x = ((src_w - 1) / 2.0 + f_src * x * scale).astype(np.float32)
    map_y = ((src_h - 1) / 2.0 + f_src * y * scale).astype(np.float32)
    return map_x, map_y


class RtspCamera(Node):
    def __init__(self):
        super().__init__("rtsp_camera")
        self.declare_parameter("url", "rtsp://192.168.144.25:8554/main.264")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 360)
        self.declare_parameter("max_fps", 15.0)
        # The camera's own delay (encode, RTSP, decode) between a frame being captured and
        # this node reading it. The lander looks up the drone attitude and gimbal angle at
        # the image stamp, so a stamp that is late puts the car in the wrong place whenever
        # the drone is pitching. Measured on flight 2026-11-13 14:42: about 0.2 s.
        self.declare_parameter("latency_s", 0.2)
        self.latency = Duration(seconds=max(0.0, float(self.get_parameter("latency_s").value)))
        self.declare_parameter("fisheye_f", 215.0)
        self.declare_parameter("out_hfov_deg", 120.0)
        self.fisheye_f = float(self.get_parameter("fisheye_f").value)
        self.out_hfov = float(self.get_parameter("out_hfov_deg").value)
        self._maps = None
        self._maps_for = None
        self.url = str(self.get_parameter("url").value)
        self.w = int(self.get_parameter("width").value)
        self.h = int(self.get_parameter("height").value)
        self.period = 1.0 / max(float(self.get_parameter("max_fps").value), 1.0)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Image, "/camera/image_raw", qos)
        self.n = 0
        self.create_timer(5.0, self._report)
        threading.Thread(target=self._loop, daemon=True).start()

    def _report(self):
        self.get_logger().info(f"/camera/image_raw: {self.n / 5.0:.1f} fps ({self.w}x{self.h})")
        self.n = 0

    def _loop(self):
        cap, last_pub, fails = None, 0.0, 0
        while rclpy.ok():
            if cap is None:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self.get_logger().warn(f"cannot open {self.url}, retrying in 2 s")
                    cap = None
                    time.sleep(2)
                    continue
                self.get_logger().info(f"camera open: {self.url}")
            ok, frame = cap.read()            # every frame is read so the buffer never backs up
            if not ok or frame is None:
                fails += 1
                if fails >= 25:
                    self.get_logger().warn("camera read failing, reopening")
                    cap.release()
                    cap, fails = None, 0
                else:
                    time.sleep(0.02)
                continue
            fails = 0
            now = time.monotonic()
            if now - last_pub < self.period:
                continue
            last_pub = now
            stamp = (self.get_clock().now() - self.latency).to_msg()     # when it was captured, not read
            if self.fisheye_f > 0.0:
                if self._maps_for != frame.shape[:2]:
                    self._maps = build_rectify_maps(frame.shape[1], frame.shape[0], self.fisheye_f,
                                                    self.w, self.h, self.out_hfov)
                    self._maps_for = frame.shape[:2]
                    self.get_logger().info(
                        f"fisheye ({self.fisheye_f:.0f} px/rad at 640 wide) -> pinhole "
                        f"{self.out_hfov:.0f} deg, {frame.shape[1]}x{frame.shape[0]} -> {self.w}x{self.h}")
                frame = cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
            elif frame.shape[1] != self.w or frame.shape[0] != self.h:
                frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
            m = Image()
            m.header.stamp = stamp
            m.header.frame_id = "camera"
            m.height, m.width = self.h, self.w
            m.encoding, m.is_bigendian, m.step = "bgr8", 0, self.w * 3
            m.data = frame.tobytes()
            self.pub.publish(m)
            self.n += 1


def main():
    rclpy.init()
    node = RtspCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
