#!/usr/bin/env python3
"""Take off and go to a preset position, height and heading, then hold.

Started by goto_stack.sh from the ground-station console ("ctrl start law=goto").
The node never arms: it waits for the PILOT to arm with the transmitter
(ardupilot_takeoff.PilotArmedTakeoff: GUIDED -> settle -> take-off to `height`),
then flies to the target at `speed` and holds it until it is stopped, the pilot
leaves GUIDED, or `max_flight_time` runs out (RTL).

    target_lat / target_lon   deg, WGS-84
    height                    m above the ground where it was armed
    yaw_deg                   compass heading to hold (0 = north, clockwise);
                              negative = keep the heading it had when armed
    speed                     m/s horizontal speed on the way (DO_CHANGE_SPEED)
    max_distance              m: refuse to fly if the target is further than this
                              from where it is armed (nothing is sent to the FCU)

The target is turned into the local frame once, at arming, from the drone's own
GPS fix and local pose, and sent as /mavros/setpoint_position/local (position +
yaw). Leaving GUIDED after arming makes the node passive for the rest of the run.
"""
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from mavros_msgs.srv import CommandLong, SetMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ardupilot_takeoff import PilotArmedTakeoff   # noqa: E402

MSG_FILE = "/tmp/auto_stack.msg"     # one line, shown in the console's status box
MAV_CMD_DO_CHANGE_SPEED = 178


def enu_offset(lat0, lon0, lat1, lon1):
    """(east, north) in metres from point 0 to point 1 on the WGS-84 ellipsoid."""
    a, e2 = 6378137.0, 6.69437999014e-3
    s = math.sin(math.radians((lat0 + lat1) / 2.0))
    w = 1.0 - e2 * s * s
    north = math.radians(lat1 - lat0) * a * (1.0 - e2) / w ** 1.5
    east = math.radians(lon1 - lon0) * a / math.sqrt(w) * math.cos(math.radians((lat0 + lat1) / 2.0))
    return east, north


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class GotoHold(Node):
    def __init__(self):
        super().__init__("goto_hold")
        dp = self.declare_parameter
        dp("target_lat", 0.0); dp("target_lon", 0.0)
        dp("height", 5.0); dp("yaw_deg", -1.0); dp("speed", 1.0)
        dp("max_flight_time", 120.0); dp("max_distance", 80.0)
        dp("arm_settle_s", 3.0); dp("armed_epoch", 0.0)
        dp("arrive_radius", 0.5)
        gp = lambda n: float(self.get_parameter(n).value)
        self.t_lat, self.t_lon = gp("target_lat"), gp("target_lon")
        self.height, self.yaw_deg, self.speed = gp("height"), gp("yaw_deg"), gp("speed")
        self.max_time, self.max_dist = gp("max_flight_time"), gp("max_distance")
        self.arrive_r = gp("arrive_radius")
        if self.t_lat == 0.0 and self.t_lon == 0.0:
            raise SystemExit("goto_hold: no target (target_lat / target_lon)")

        self.fix = None            # (lat, lon, t)
        self.pose = None           # (x, y, z)
        self.compass = None        # deg
        self.target = None         # (x, y, z) local ENU, set at arming
        self.yaw_enu = 0.0
        self.passive = None        # reason, once the node has stopped commanding
        self.rtl_sent = False
        self.arrived = False
        self._msg = ""

        self.create_subscription(NavSatFix, "/mavros/global_position/global", self._on_fix, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self._on_pose, qos_profile_sensor_data)
        self.create_subscription(Float64, "/mavros/global_position/compass_hdg", self._on_compass, qos_profile_sensor_data)
        self.sp_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        self.cmd_client = self.create_client(CommandLong, "/mavros/cmd/command")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")

        self.takeoff = PilotArmedTakeoff(
            self, altitude=self.height, on_armed=self._on_armed, on_airborne=self._on_airborne,
            settle_s=gp("arm_settle_s"), armed_epoch=gp("armed_epoch"))
        self.create_timer(0.2, self._tick)        # 5 Hz setpoints
        self.create_timer(1.0, self._report)
        yaw = f"{self.yaw_deg:.0f} deg" if self.yaw_deg >= 0 else "as armed"
        self.get_logger().info(
            f"goto_hold: target {self.t_lat:.7f},{self.t_lon:.7f}, {self.height:.1f} m up, heading {yaw}, "
            f"{self.speed:.1f} m/s, RTL after {self.max_time:.0f} s, limit {self.max_dist:.0f} m")

    # ---------------------------------------------------------------- inputs
    def _on_fix(self, m):
        if m.status.status >= 0 and m.latitude != 0.0:
            self.fix = (m.latitude, m.longitude, time.monotonic())

    def _on_pose(self, m):
        p = m.pose.position
        self.pose = (p.x, p.y, p.z)

    def _on_compass(self, m):
        self.compass = float(m.data)

    # ------------------------------------------------------- take-off hooks
    def _refuse(self, why):
        """Before take-off: stop the sequence. Nothing has been sent to the FCU."""
        self.passive = why
        self.takeoff.cancel()
        self.get_logger().error(f"REFUSED: {why}. Nothing sent to the FCU - disarm.")

    def _on_armed(self):
        if self.fix is None or time.monotonic() - self.fix[2] > 2.0:
            return self._refuse("no fresh GPS position from the FCU")
        if self.pose is None:
            return self._refuse("no local position from the FCU")
        east, north = enu_offset(self.fix[0], self.fix[1], self.t_lat, self.t_lon)
        dist = math.hypot(east, north)
        if dist > self.max_dist:
            return self._refuse(f"target is {dist:.0f} m away, limit {self.max_dist:.0f} m")
        if self.yaw_deg >= 0.0:
            compass = self.yaw_deg
        elif self.compass is not None:
            compass = self.compass
        else:
            return self._refuse("no compass heading to keep (give a heading)")
        self.yaw_enu = math.pi / 2.0 - math.radians(compass)
        self.hold_compass = compass % 360.0
        self.target = (self.pose[0] + east, self.pose[1] + north, self.pose[2] + self.height)
        self.get_logger().info(
            f"Target is {dist:.1f} m away ({east:+.1f} m east, {north:+.1f} m north), "
            f"{self.height:.1f} m above here, heading {self.hold_compass:.0f} deg")

    def _on_airborne(self):
        self.t_air = time.monotonic()
        req = CommandLong.Request()
        req.command = MAV_CMD_DO_CHANGE_SPEED
        req.param1, req.param2, req.param3 = 1.0, float(self.speed), -1.0   # ground speed, m/s, throttle unchanged
        if self.cmd_client.service_is_ready():
            self.cmd_client.call_async(req).add_done_callback(self._speed_done)
        else:
            self.get_logger().warn("command service not ready: speed not set, FCU default speed applies")
        self.get_logger().info("Take-off complete: going to the target.")

    def _speed_done(self, fut):
        try:
            ok = bool(fut.result().success)
        except Exception as e:
            ok = False
            self.get_logger().error(f"speed command exception: {e}")
        if ok:
            self.get_logger().info(f"Speed set to {self.speed:.1f} m/s")
        else:
            self.get_logger().warn("FCU did not accept the speed: its default speed applies")

    # ----------------------------------------------------------------- loop
    def _tick(self):
        if not self.takeoff.airborne or self.passive or self.target is None:
            return
        st = self.takeoff._state
        if st is None or not st.armed:
            self.passive = "disarmed"
            return
        if st.mode != "GUIDED":
            if not self.rtl_sent:
                self.get_logger().warn(f"FCU is in {st.mode}, not GUIDED: the pilot has control. "
                                       "Passive for the rest of this run.")
            self.passive = f"pilot has control ({st.mode})"
            return
        if time.monotonic() - self.t_air >= self.max_time:
            self._rtl()
            return
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = self.target
        m.pose.orientation.z = math.sin(self.yaw_enu / 2.0)
        m.pose.orientation.w = math.cos(self.yaw_enu / 2.0)
        self.sp_pub.publish(m)

    def _rtl(self):
        self.rtl_sent = True
        self.passive = f"{self.max_time:.0f} s up: RTL"
        self.get_logger().warn(f"{self.max_time:.0f} s since take-off - requesting RTL")
        req = SetMode.Request()
        req.custom_mode = "RTL"
        if self.mode_client.service_is_ready():
            self.mode_client.call_async(req)
        else:
            self.get_logger().error("SetMode service not available for RTL - take over")

    def _report(self):
        if self.passive:
            text = f"goto: {self.passive}"
        elif not self.takeoff.airborne:
            text = f"goto: {self.takeoff.status}"
        else:
            d = math.hypot(self.target[0] - self.pose[0], self.target[1] - self.pose[1])
            dz = self.pose[2] - self.target[2]
            left = self.max_time - (time.monotonic() - self.t_air)
            yaw = "" if self.compass is None else f", heading {self.compass:.0f} (want {self.hold_compass:.0f})"
            if d <= self.arrive_r and not self.arrived:
                self.arrived = True
                self.get_logger().info(f"At the target ({d:.2f} m, height error {dz:+.2f} m): holding.")
            text = (f"goto: {'holding at' if d <= self.arrive_r else 'flying to'} target, {d:.1f} m to go, "
                    f"height {dz:+.1f} m{yaw}, RTL in {left:.0f} s")
        if text != self._msg:
            self._msg = text
            try:
                with open(MSG_FILE, "w") as fh:
                    fh.write(text + "\n")
            except OSError:
                pass


def main():
    rclpy.init()
    node = GotoHold()
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
