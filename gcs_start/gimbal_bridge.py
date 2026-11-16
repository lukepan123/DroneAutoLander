#!/usr/bin/env python3
"""auto_lander gimbal commands <-> SIYI A2 mini, on the Pi.

The lander's gimbal_controller talks to the gimbal through ArduPilot's mount
driver (MAVROS gimbal_control plugin). On this drone the flight controller has no
mount (MNT1_TYPE 0) and no wire to the gimbal: the A2 mini is on the Pi's
ethernet and only answers the SIYI SDK. This node stands in for that path:

    /mavros/gimbal_control/manager/set_pitchyaw   (GimbalManagerSetPitchyaw, pitch in rad
            as gimbal_controller sends it)  ->  SIYI rate commands, closed on the
            measured angle at 20 Hz
    SIYI attitude  ->  /mavros/gimbal_control/device/attitude_status
            (GimbalDeviceAttitudeStatus; gimbal_controller reads the pitch out of q)

Why a rate loop and not the SIYI "set angle" command: in the A2's auto-detected
upside-down mount mode that command's pitch sign is inverted and its soft limits
sit on the wrong side (bench, 2026-09-10, ground_station/a2_gui.py). The rate
command is not flipped and the attitude read-back is usable, so the loop is
    error = target - measured  ->  rate = K * error (clamped)  ->  0x07

Angle frames. The lander treats the gimbal pitch as relative to the drone BODY
(0 = along the body x axis, -90 = straight down), which is how its SITL servo
behaves. The A2 stabilises itself and reports pitch against GRAVITY. With
body_frame:=true (default) this node converts both ways with the drone's own
pitch from /mavros/imu/data:
    body pitch = gravity pitch - drone nose-up pitch
so the lander's camera-pose maths does not count the drone's pitch twice. Without
an IMU message newer than 0.5 s the drone pitch is taken as 0 (logged).

This node must be the ONLY SIYI SDK client while it runs: the A2 drops replies and
commands when several programs poll it (recorder.py was also polling at 10 Hz in the
flights of 2026-11-13). It writes each reading to /tmp/gimbal_bridge.state and
recorder.py logs from that file instead of asking the gimbal itself.

The MAVROS gimbal_control plugin must not be loaded (it would publish/forward on
the same topics); goto_stack.sh's plugin list leaves it out.
"""
import math
import os
import socket
import struct
import sys
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from mavros_msgs.msg import GimbalDeviceAttitudeStatus, GimbalManagerSetPitchyaw

sys.path.insert(0, os.path.expanduser("~"))
import siyi   # noqa: E402  frame builder / CRC / command ids (/home/case/siyi.py)

LOOP_HZ = 20.0
# deg -> % rate. a2_gui.py used K 3 / max 100 for a slider on the bench; in flight on
# 2026-11-13 that swept the gimbal stop to stop (-85 <-> +26 deg at ~60 deg/s, every few
# seconds). Half the gain and half the top rate leave room for a late or lost packet.
K, RATE_MIN, RATE_MAX, DEADBAND = 2.0, 6, 50, 0.6
# Feed-forward of the target's own rate, so a moving target is followed without a
# standing error (bench: 100 % is about 57 deg/s, so 1 % is about 0.57 deg/s).
DEG_S_PER_PCT, FF_MAX, FF_ALPHA = 0.57, 35.0, 0.3
STATE_FILE = "/tmp/gimbal_bridge.state"               # "t_monotonic gravity_pitch raw_pitch": read by recorder.py
PITCH_MIN, PITCH_MAX = -85.0, 25.0                    # measured stops, gravity frame
STALL_S, STALL_DEG = 1.0, 0.8
CMD_TIMEOUT = 1.0                                     # s without a command: stop moving


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class A2:
    """Fire-and-forget rate + polled attitude on one UDP socket."""

    def __init__(self, host, port):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.15)
        self.seq = 0
        self.upside_down = None
        self.raw = float("nan")

    def _tx(self, cmd, data=b""):
        self.seq = (self.seq + 1) & 0xFFFF
        self.sock.sendto(siyi.build_frame(cmd, data, self.seq), self.addr)

    def _ask(self, cmd, data=b""):
        self._tx(cmd, data)
        end = time.time() + 0.15
        while time.time() < end:
            try:
                buf, _ = self.sock.recvfrom(1024)
            except (socket.timeout, OSError):
                return None
            p = siyi.parse_frame(buf)
            if p and p[0] == cmd:
                return p[1]
        return None

    def mount_dir(self):
        for _ in range(4):
            d = self._ask(siyi.CMD_GIMBAL_INFO)
            if d:
                return d[5] if len(d) >= 6 else d[3] if len(d) >= 4 else None   # A2 mini: 5 bytes, flag at [3]
        return None

    def pitch(self):
        """Gravity-referenced pitch, 0 level / -90 down, or None."""
        d = self._ask(siyi.CMD_ATTITUDE)
        if not d or len(d) < 12:
            return None
        self.raw = struct.unpack("<hhhhhh", d[:12])[1] / 10.0
        return wrap(180.0 - self.raw) if self.upside_down else self.raw

    def rate(self, pct):
        self._tx(siyi.CMD_GIMBAL_SPEED, struct.pack("<bb", 0, int(max(-100, min(100, pct)))))


class GimbalBridge(Node):
    def __init__(self):
        super().__init__("gimbal_bridge")
        self.declare_parameter("host", siyi.DEFAULT_HOST)
        self.declare_parameter("port", siyi.DEFAULT_PORT)
        self.declare_parameter("body_frame", True)
        self.body_frame = bool(self.get_parameter("body_frame").value)
        self.g = A2(str(self.get_parameter("host").value), int(self.get_parameter("port").value))
        md = self.g.mount_dir()
        self.g.upside_down = md == 2
        self.get_logger().info(f"A2 mini mount mode {md} ({'upside down' if self.g.upside_down else 'normal'}); "
                               f"angles {'relative to the drone body' if self.body_frame else 'against gravity'}")

        self.lock = threading.Lock()
        self.cmd = None            # (pitch deg in the lander's frame, t)
        self.cmd_rate = 0.0        # deg/s the commanded pitch is moving at (smoothed)
        self.drone_pitch = 0.0     # deg, nose up positive
        self.imu_t = 0.0
        self._imu_warned = False
        self.n_meas = self.n_miss = 0
        self.last = (None, None, 0)   # measured (lander frame), target, rate

        self.pub = self.create_publisher(GimbalDeviceAttitudeStatus,
                                         "/mavros/gimbal_control/device/attitude_status", 10)
        self.create_subscription(GimbalManagerSetPitchyaw,
                                 "/mavros/gimbal_control/manager/set_pitchyaw", self._on_cmd, 10)
        self.create_subscription(Imu, "/mavros/imu/data", self._on_imu, qos_profile_sensor_data)
        self.create_timer(5.0, self._report)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- inputs
    def _on_cmd(self, m):
        if math.isfinite(m.pitch):
            now, deg = time.monotonic(), math.degrees(m.pitch)
            with self.lock:
                if self.cmd is not None and 0.005 < now - self.cmd[1] < 0.5:
                    inst = (deg - self.cmd[0]) / (now - self.cmd[1])
                    self.cmd_rate += FF_ALPHA * (inst - self.cmd_rate)      # deg/s, smoothed
                else:
                    self.cmd_rate = 0.0
                self.cmd = (deg, now)

    def _on_imu(self, m):
        q = m.orientation
        # world z of the body x axis (ENU, z up) = sine of the nose-up pitch
        fz = 2.0 * (q.x * q.z - q.w * q.y)
        with self.lock:
            self.drone_pitch = math.degrees(math.asin(max(-1.0, min(1.0, fz))))
            self.imu_t = time.monotonic()

    def _drone_pitch(self):
        if not self.body_frame:
            return 0.0
        with self.lock:
            fresh = time.monotonic() - self.imu_t < 0.5
            p = self.drone_pitch
        if not fresh:
            if not self._imu_warned:
                self._imu_warned = True
                self.get_logger().warn("no fresh /mavros/imu/data: taking the drone's pitch as 0")
            return 0.0
        self._imu_warned = False
        return p

    # ------------------------------------------------------------------ loop
    def _loop(self):
        last_good = None
        push_since = push_from = None
        stall_target = None
        while not self._stop.is_set():
            t0 = time.monotonic()
            grav = self.g.pitch()                      # gravity frame
            if grav is None:
                self.n_miss += 1
                if last_good and t0 - last_good[0] < 0.3:
                    grav = last_good[1]                # ride through one miss
            else:
                self.n_meas += 1
                last_good = (t0, grav)
            dp = self._drone_pitch()
            with self.lock:
                cmd, cmd_rate = self.cmd, self.cmd_rate
            tgt_grav = None
            if cmd is not None and t0 - cmd[1] < CMD_TIMEOUT:
                tgt_grav = max(PITCH_MIN, min(PITCH_MAX, cmd[0] + dp))
            rate = 0
            if stall_target is not None and (tgt_grav is None or abs(tgt_grav - stall_target) > 1.0):
                stall_target = None                    # a new target clears a stall
            if grav is not None and tgt_grav is not None and stall_target is None:
                err = tgt_grav - grav
                ff = max(-FF_MAX, min(FF_MAX, cmd_rate / DEG_S_PER_PCT)) if abs(cmd_rate) > 2.0 else 0.0
                if PITCH_MIN < tgt_grav < PITCH_MAX and ff:
                    rate = max(-RATE_MAX, min(RATE_MAX, K * err + ff))       # following a moving target
                    if abs(rate) < RATE_MIN:
                        rate = 0
                    push_since = None
                elif abs(err) > DEADBAND:
                    rate = max(RATE_MIN, min(RATE_MAX, K * abs(err))) * (1 if err > 0 else -1)
                    if push_since is None:
                        push_since, push_from = t0, grav
                    elif abs(grav - push_from) >= STALL_DEG:
                        push_since, push_from = t0, grav
                    elif t0 - push_since > STALL_S:    # pushed and nothing moved: at a stop
                        stall_target, rate = tgt_grav, 0
                else:
                    push_since = None
            if not rate:
                push_since = None
            # The A2 never acknowledges a rate command and keeps turning at the last rate
            # it heard, so the rate goes out EVERY cycle, zero included: one lost "stop"
            # packet must not leave it running into a stop.
            self.g.rate(rate)
            if grav is not None:
                try:                                   # this node is the only SDK client:
                    with open(STATE_FILE + ".tmp", "w") as fh:   # recorder.py logs from here
                        fh.write(f"{t0:.3f} {grav:.2f} {self.g.raw:.2f}\n")
                    os.replace(STATE_FILE + ".tmp", STATE_FILE)
                except OSError:
                    pass
            if grav is not None:
                body = grav - dp
                m = GimbalDeviceAttitudeStatus()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = "gimbal"
                half = math.radians(body) / 2.0
                m.q.x, m.q.y, m.q.z, m.q.w = 0.0, math.sin(half), 0.0, math.cos(half)
                try:
                    self.pub.publish(m)
                except Exception:              # ROS is shutting down under us
                    break
                self.last = (body, None if tgt_grav is None else tgt_grav - dp, rate)
            time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - t0)))
        for _ in range(5):                     # leave it stationary, whatever gets lost
            self.g.rate(0)
            time.sleep(0.02)
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass

    def _report(self):
        meas, tgt, rate = self.last
        self.get_logger().info(
            f"pitch {'--' if meas is None else '%.1f' % meas} deg, target "
            f"{'none' if tgt is None else '%.1f' % tgt}, rate {rate:+.0f} %, drone pitch {self._drone_pitch():+.1f}; "
            f"{self.n_meas / 5.0:.0f} Hz read-back, {self.n_miss} missed")
        self.n_meas = self.n_miss = 0

    def shutdown(self):
        self._stop.set()
        self.thread.join(timeout=1.0)
        try:
            self.g.rate(0)
        except Exception:
            pass


def main():
    rclpy.init()
    node = GimbalBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()               # always leaves the gimbal with rate 0
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
