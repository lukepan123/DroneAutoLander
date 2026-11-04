#!/usr/bin/env python3
"""
Static perception sweep for auto_lander (Gazebo + ArduPilot SITL).

Per case (slant range R, elevation e, aspect psi, all relative to the pad):
  1. moves the drone (GUIDED, MAVLink) to the case altitude, holding x/y and yaw
  2. teleports the stationary car so the pad sits at the right horizontal distance
     along the camera heading, rotated to the requested aspect
  3. waits SETTLE_S of sim time (gimbal re-centres, car settles), then records a
     LOG_S sim-time window in manifest.csv

The YOLO box fit is computed and logged by your controller node, launched with
AUTO_LANDER_SWEEP=1 so it never commands the FCU. analyse_sweep.py cuts its controller_*.csv
into these windows and scores the raw fit only.

Re-running skips cases already marked ok, so a crash costs at most one case.

Usage (normally called by run_sweep.sh):
  python3 sweep_runner.py --run-dir sweep_out/run_YYYYmmdd_HHMMSS [--limit 3]
"""
import os
# The apt gz-msgs bindings were generated for an older protobuf than the pip one in
# ~/.local. The pure-Python backend reads them fine; slower, but only for this script.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import csv
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

from pymavlink import mavutil

try:                                    # Gazebo Harmonic
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
except ImportError:                     # Gazebo Jetty
    from gz.transport14 import Node
    from gz.msgs11.pose_v_pb2 import Pose_V

# ------------------------------------------------------------------ config
WORLD          = "iris_runway_new"      # <world name="..."> (matches your gz_bridge topics)
CAR_MODEL      = "LandingVehicle"         # check with: gz model --list
DRONE_MODEL    = "iris_with_gimbal"
MAV_URL        = "udp:127.0.0.1:14556"  # second SITL --out from run_sweep.sh; MAVROS keeps 14555

RANGES_M       = [2.5, 5, 10, 15, 20]
ELEV_DEG       = [15, 30, 45, 60, 75, 90]
ASPECT_DEG     = [0, 30, 60, 90, 120, 150, 180]  # 0 = camera looking at the car's nose
ASPECT_NADIR   = [0, 90, 180]                          # used when elev = 90

# Pad top-centre in the car frame (x fwd, y left, z up): keep equal to yolo_car_pad_position
PAD_OFFSET_XY  = (-0.5, 0.0)
PAD_HEIGHT_M   = 1.525
CAR_Z          = 0.05    # small drop so the tyres don't spawn inside the ground
CAR_YAW_OFFSET = 0.0     # rad, if the car model's forward axis is not +x
MIN_ALT_M      = 3.0     # skip geometries that would put the drone below this AGL
HOLD_YAW_DEG   = 0.0     # drone heading held throughout (NED, from north)

SETTLE_S       = 4.0     # sim s after each teleport before logging (transient is gone in <1.5 s)
FIRST_SETTLE_S = 8.0     # first case only: box-fit roof height converges after startup
LOG_S          = 3.0     # sim s logged per case (~30 fits; frame-to-frame noise is mm-level)
ALT_TOL_M      = 0.3
VEL_TOL        = 0.25    # m/s
STABLE_S       = 1.0     # must stay inside tolerance this long (wall s)
REACH_TIMEOUT  = 60.0    # wall s

OUT_DIR        = Path(os.environ.get("SWEEP_OUT",                     # SWEEP-TEMP: per condition
                                 Path(__file__).resolve().parent / "sweep_out"))
# -------------------------------------------------------------------------

FIELDS = ["case_id", "R", "elev", "aspect", "alt", "d", "run_dir",
          "t_start", "t_end", "status"]


# ----------------------------------------------------------------- Gazebo
def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


class GzPoses:
    """Latest drone/car pose and sim time from Gazebo."""

    def __init__(self):
        self.node = Node()
        self.lock = threading.Lock()
        self.latest, self.t = {}, 0.0
        topic = f"/world/{WORLD}/pose/info"
        if not self.node.subscribe(Pose_V, topic, self._cb):
            sys.exit(f"Could not subscribe to {topic}")

    def _cb(self, msg):
        got = {p.name: (p.position.x, p.position.y, p.position.z, yaw_from_quat(p.orientation))
               for p in msg.pose if p.name in (DRONE_MODEL, CAR_MODEL)}
        with self.lock:
            self.t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            self.latest.update(got)

    def now(self):
        with self.lock:
            return self.t

    def pose(self, name):
        with self.lock:
            return self.latest.get(name)

    def wait_sim(self, dt):
        """Wait in SIM time, so results hold at any real-time factor."""
        t_end = self.now() + dt
        while self.now() < t_end:
            time.sleep(0.02)

    def wait_ready(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if DRONE_MODEL in self.latest and CAR_MODEL in self.latest:
                    return
            time.sleep(0.2)
        sys.exit("No Gazebo poses for drone/car: check WORLD, CAR_MODEL, DRONE_MODEL")


def set_car_pose(x, y, yaw):
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    req = (f'name: "{CAR_MODEL}" position {{x: {x:.4f} y: {y:.4f} z: {CAR_Z:.4f}}} '
           f'orientation {{x: 0 y: 0 z: {qz:.6f} w: {qw:.6f}}}')
    r = subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD}/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "2000", "--req", req],
        capture_output=True, text=True)
    if "true" not in r.stdout:
        raise RuntimeError(f"set_pose failed: {r.stdout.strip()} {r.stderr.strip()}")


def place_car(gz, c):
    """Pad at horizontal distance d along the camera heading; car rotated to the aspect."""
    dx, dy, _, dyaw = gz.pose(DRONE_MODEL)
    px, py = dx + c["d"] * math.cos(dyaw), dy + c["d"] * math.sin(dyaw)
    car_yaw = dyaw + math.pi + math.radians(c["aspect"]) + CAR_YAW_OFFSET
    ox, oy = PAD_OFFSET_XY
    cx = px - (math.cos(car_yaw) * ox - math.sin(car_yaw) * oy)
    cy = py - (math.sin(car_yaw) * ox + math.cos(car_yaw) * oy)
    set_car_pose(cx, cy, car_yaw)


# ---------------------------------------------------------------- MAVLink
class Drone:
    def __init__(self):
        self.m = mavutil.mavlink_connection(MAV_URL)
        self.m.wait_heartbeat()
        self.cmd(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                 mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 1e6 / 20)

    def cmd(self, command, *params):
        p = list(params) + [0] * (7 - len(params))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     command, 0, *p)

    def drain(self):
        while self.m.recv_match(blocking=False) is not None:
            pass

    def fresh(self, mtype, timeout=2):
        self.drain()
        return self.m.recv_match(type=mtype, blocking=True, timeout=timeout)

    def takeoff(self, alt):
        t0 = time.time()
        while not self.m.motors_armed():
            if time.time() - t0 > 180:
                sys.exit("Arming failed: check pre-arm messages in the SITL log")
            self.m.set_mode("GUIDED")
            self.m.arducopter_arm()
            self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        self.cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)
        return self.wait_alt(-alt)

    def goto(self, x, y, z):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b100111111000,                       # position + yaw only
            x, y, z, 0, 0, 0, 0, 0, 0, math.radians(HOLD_YAW_DEG), 0)

    def wait_alt(self, z):
        self.drain()
        t0, stable = time.time(), None
        while time.time() - t0 < REACH_TIMEOUT:
            m = self.m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1)
            if m is None:
                continue
            ok = (abs(m.z - z) < ALT_TOL_M
                  and math.sqrt(m.vx ** 2 + m.vy ** 2 + m.vz ** 2) < VEL_TOL)
            if not ok:
                stable = None
            elif stable is None:
                stable = time.time()
            elif time.time() - stable > STABLE_S:
                return True
        return False

    def land(self):
        self.m.set_mode("LAND")


# ------------------------------------------------------------------ sweep
def build_cases():
    """Order cases so the camera's view changes gradually (serpentine through the grid).

    Elevation is the outer loop, starting at nadir to match the gimbal's initial -90 deg.
    Within one elevation the line of sight keeps the same depression angle, so as R grows
    the car just moves further out along it and the gimbal barely moves. Range then
    reverses direction on the next elevation (5 -> 40 m, then 40 -> 5 m, ...), so each
    elevation change keeps the same R and only swings the gimbal by 15 deg. Aspects
    alternate direction too, so the car never spins back through 180 deg in one step.
    """
    cases, r_up, a_up = [], True, True
    for e in sorted(ELEV_DEG, reverse=True):
        ranges = [R for R in RANGES_M
                  if PAD_HEIGHT_M + R * math.sin(math.radians(e)) >= MIN_ALT_M]
        for R in (ranges if r_up else ranges[::-1]):
            alt = PAD_HEIGHT_M + R * math.sin(math.radians(e))
            d = 0.0 if e == 90 else R * math.cos(math.radians(e))
            aspects = ASPECT_NADIR if e == 90 else ASPECT_DEG
            for a in (aspects if a_up else aspects[::-1]):
                cases.append(dict(case_id=f"R{R}_E{e}_A{a}", R=R, elev=e, aspect=a,
                                  alt=round(alt, 3), d=round(d, 3)))
            a_up = not a_up
        r_up = not r_up
    return cases


def load_done(path):
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {r["case_id"] for r in csv.DictReader(f) if r["status"] == "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="folder holding this launch's controller CSV")
    ap.add_argument("--limit", type=int, help="run only the first N pending cases")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man_path = OUT_DIR / "manifest.csv"
    cases = build_cases()
    done = load_done(man_path)
    todo = [c for c in cases if c["case_id"] not in done][:args.limit]
    if not todo:
        print("Nothing to do: all cases complete.")
        return
    print(f"{len(todo)} of {len(cases)} cases pending, "
          f"about {len(todo) * (SETTLE_S + LOG_S) / 60:.0f} sim-min plus climbs")

    gz = GzPoses()
    gz.wait_ready()
    drone = Drone()

    try:
        hover = todo[0]["alt"]                # take off straight to the first case
        if not drone.takeoff(hover):
            sys.exit("Takeoff did not settle")
        lp = drone.fresh("LOCAL_POSITION_NED")
        x0, y0 = lp.x, lp.y
        drone.goto(x0, y0, -hover)                # also applies HOLD_YAW_DEG
        drone.wait_alt(-hover)

        new = not man_path.exists()
        with open(man_path, "a", newline="") as fm:
            man = csv.DictWriter(fm, FIELDS)
            if new:
                man.writeheader()

            cur_alt = None
            for i, c in enumerate(todo, 1):
                row = {k: c[k] for k in ("case_id", "R", "elev", "aspect", "alt", "d")}
                row["run_dir"] = args.run_dir
                try:
                    if c["alt"] != cur_alt:
                        drone.goto(x0, y0, -c["alt"])
                        if not drone.wait_alt(-c["alt"]):
                            raise RuntimeError("altitude not reached")
                        cur_alt = c["alt"]
                    place_car(gz, c)
                    gz.wait_sim(FIRST_SETTLE_S if i == 1 else SETTLE_S)
                    row["t_start"] = gz.now()
                    gz.wait_sim(LOG_S)
                    row["t_end"] = gz.now()
                    row["status"] = "ok"
                except Exception as e:                  # log and move on
                    row["status"] = f"failed: {e}"
                    cur_alt = None
                man.writerow(row)
                fm.flush()
                print(f"[{i}/{len(todo)}] {c['case_id']}: {row['status']}", flush=True)
    finally:
        drone.land()


if __name__ == "__main__":
    main()