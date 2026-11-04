#!/usr/bin/env bash
# =============================================================================
#  run_batch.sh : unattended SITL sweep for auto_lander
#
#  For every (repeat, trajectory, speed):
#    1. start Gazebo, ArduPilot SITL and MAVROS (logged, no terminal tabs)
#    2. wait for "EKF3 IMU0 is using GPS" and a MAVROS heartbeat
#    3. ros2 launch auto_lander with trajectory/speed as launch arguments
#    4. watch the controller CSV until controller_state >= 6000 (landed / RTL)
#    5. shut the launch down cleanly, kill the sim, file every output, log a row
#
#  Usage   : ./run_batch.sh [study_name]          (default: main)
#  Subset  : override any of TRAJECTORIES, SPEEDS, REPEATS, WORLD from the shell, e.g.
#            SPEEDS="0" TRAJECTORIES="straight" WORLD=iris_runway_static.sdf ./run_batch.sh main
#  MBZIRC  : TRAJECTORIES="mbzirc" SPEEDS="15" ./run_batch.sh mbzirc
#            Each run gets its own world (run dir/world.sdf, from MBZIRC_WORLD) with the
#            car spawned at that repeat's lap phase; WORLD is not used for mbzirc runs.
#  Swept   : straight / circle / sin runs above 0 km/h also get their own world (from WORLD)
#            with the car spawned d_ramp(v) behind the cruise line (agv_controller
#            swept_world_poses), so it is at cruise before the UAV can see it.
#            Use a NEW study name after this change: old runs carry DONE files and would be
#            skipped, mixing the two lead-in protocols.
#  Resume  : run the same command again; finished runs are skipped
#  Re-run  : delete that run's folder (or its DONE file)
#  Stop    : Ctrl+C (the sim is torn down on exit)
# =============================================================================
set -o pipefail
set -m                      # job control: background jobs keep SIGINT, so launch can shut down cleanly

# ---------------- Study design ----------------
read -ra TRAJECTORIES <<< "${TRAJECTORIES:-straight circle sin}"
read -ra SPEEDS_KMH   <<< "${SPEEDS:-$(seq -s ' ' 5 5 70)}"     # 0 km/h run separately (own world)
REPEATS=${REPEATS:-10}
WORLD=${WORLD:-iris_runway_new.sdf}   # file in ~/ardupilot_gazebo/worlds; keep <world name="iris_runway_new">
ZERO_SPEED_ONCE=1           # 0 km/h is the same run for every trajectory: only do it under the first
ACCEL_DISTANCE_M=10.0       # mbzirc only: its ramp from rest. Swept runs ramp at the Prius limits
TURN_START_DISTANCE_M=35.0  # straight cruise past the cruise line before any turn (35 m = turn-in
                            # abeam the UAV waiting point; keep equal to agv_controller default)
SIN_PHASE_STRATA=${SIN_PHASE_STRATA:-10}   # sine start phase = ((rep-1) % N) * 360/N deg, same at
                                           # every speed, so each speed sees the same phase set.
                                           # Keep N = REPEATS; 0 = always start at phase 0.
                                           # mbzirc reuses it as the lap phase the car spawns at
MBZIRC_WORLD=${MBZIRC_WORLD:-iris_mbzirc.sdf}  # template in ~/ardupilot_gazebo/worlds for mbzirc runs
MBZIRC_AGV_DELAY_S=${MBZIRC_AGV_DELAY_S:-5.0}  # launch -> car start for mbzirc (others keep the launch
                                               # default, 25 s). The car was already lapping in the
                                               # competition; keep >= ~5 s so the gz bridge is up first
MBZIRC_GPS_LAT=${MBZIRC_GPS_LAT:--35.365605}   # mbzirc waiting point = centre of the figure-8,
MBZIRC_GPS_LON=${MBZIRC_GPS_LON:-149.164137}   # world (-100, -260); others keep the launch default

# ---------------- Paths ----------------
WS="$HOME/ros2_ws"
ARDUPILOT="$HOME/ardupilot"
ARDU_GZ="$HOME/ardupilot_gazebo"
STUDY="${1:-main}"
RESULTS="$HOME/sitl_results/$STUDY"

# ---------------- Timing (s) ----------------
EKF_TIMEOUT=180             # sim start -> EKF on GPS + MAVROS connected
RUN_TIMEOUT=300             # launch -> terminal state (controller max_runtime is 200 s after takeoff)
TERMINAL_HOLD=3             # state must sit >= 6000 this long before teardown
MAX_ATTEMPTS=3              # retries when the sim fails to boot or a run hangs

# ---------------- Display ----------------
SHOW_GUI=0                  # 0 = Gazebo server only + no MAVProxy console/map (test once first)

if (( SHOW_GUI )); then
    GZ_FLAGS="";                       SITL_UI="--console --map"
else
    GZ_FLAGS="-s --headless-rendering"; SITL_UI=""
fi

export PYTHONUNBUFFERED=1               # so MAVProxy / launch output reaches the logs live
export RCUTILS_LOGGING_BUFFERED_STREAM=0

# =============================================================================
log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$RESULTS/batch.log"; }

ros_env() { source /opt/ros/humble/setup.bash; source "$WS/install/setup.bash"; }

SIM_PATTERN='lib/auto_lander/|parameter_bridge|mavros_node|sim_vehicle.py|mavproxy|arducopter|gz sim'

kill_sim() {
    pkill -INT -f "ros2 launch auto_lander" 2>/dev/null
    sleep 2
    pkill -f "$SIM_PATTERN" 2>/dev/null
    sleep 3
    pkill -9 -f "$SIM_PATTERN" 2>/dev/null
    sleep 1
}

wait_pid() {    # $1 = pid, $2 = max seconds
    local pid=$1 t=$2
    while kill -0 "$pid" 2>/dev/null && (( t-- > 0 )); do sleep 1; done
}

start_sim() {   # $1 = run dir (logs land here)
    local d="$1"

    (
        export GZ_SIM_SYSTEM_PLUGIN_PATH="$ARDU_GZ/build:$GZ_SIM_SYSTEM_PLUGIN_PATH"
        export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
        cd "$ARDU_GZ/worlds" && exec gz sim $GZ_FLAGS "$RUN_WORLD" -v -r
    ) </dev/null >"$d/gazebo.log" 2>&1 &
    sleep 5

    (
        cd "$ARDUPILOT" && exec "$ARDUPILOT/Tools/autotest/sim_vehicle.py" -v ArduCopter $SITL_UI \
            -f JSON \
            --add-param-file="$ARDU_GZ/config/gazebo-iris-gimbal_1d.parm" \
            --out=udp:127.0.0.1:14555 \
            --mavproxy-args="--non-interactive --cmd=\"set streamrate 40\""
    ) </dev/null >"$d/sitl.log" 2>&1 &
    sleep 10

    (
        ros_env
        exec ros2 run mavros mavros_node --ros-args \
            -p use_sim_time:=true \
            -p fcu_url:=udp://:14555@ \
            --params-file "$(ros2 pkg prefix mavros)/share/mavros/launch/apm_config.yaml" \
            --params-file "$(ros2 pkg prefix mavros)/share/mavros/launch/apm_pluginlists.yaml" \
            -p send.tf:=true
    ) </dev/null >"$d/mavros.log" 2>&1 &
}

make_world() {   # $1 = run dir, $2 = trajectory, $3 = speed (km/h), $4 = lap phase (deg)
                 # -> writes $1/world.sdf, prints car pose
    # Poses come from agv_controller itself, so the spawn always matches the trajectory
    local template="$WORLD"
    [[ $2 == mbzirc ]] && template="$MBZIRC_WORLD"
    (
        ros_env
        python3 - "$ARDU_GZ/worlds/$template" "$1/world.sdf" "$2" "$3" "$4" <<'PY'
import re, sys
from auto_lander.agv_controller import (KMH_TO_MS, UAV_WAIT_WORLD, mbzirc_world_poses,
                                        swept_world_poses)

src, dst, traj, speed_kmh, phase = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
sdf = open(src).read()

def get_pose(sdf, uri):
    """(x, y, yaw) of the first <pose> after <uri>model://uri</uri>."""
    m = re.search(r'<uri>model://%s</uri>.*?<pose degrees="true">\s*(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s*</pose>'
                  % re.escape(uri), sdf, re.S)
    if not m:
        sys.exit(f"no model://{uri} pose found in {src}")
    return tuple(float(v) for v in m.groups())

def set_pose(sdf, uri, pose):
    """Replace x, y and yaw of the first <pose> after <uri>model://uri</uri>; keep z, roll, pitch."""
    pat = re.compile(r'(<uri>model://%s</uri>.*?<pose degrees="true">)\s*\S+\s+\S+\s+(\S+\s+\S+\s+\S+)\s+\S+\s*(</pose>)'
                     % re.escape(uri), re.S)
    out, n = pat.subn(lambda m: f"{m.group(1)}{pose[0]:.3f} {pose[1]:.3f} {m.group(2)} {pose[2]:.3f}{m.group(3)}",
                      sdf, count=1)
    if n != 1:
        sys.exit(f"no model://{uri} pose found in {src}")
    return out

if traj == "mbzirc":
    car, uav = mbzirc_world_poses(phase)
    sdf = set_pose(sdf, "iris_with_gimbal", uav)
else:
    # Swept runs: only the car moves. The layout is relative to the UAV waiting point, so
    # refuse to run if the template's UAV is not where agv_controller assumes it is.
    ux, uy, _ = get_pose(sdf, "iris_with_gimbal")
    if abs(ux - UAV_WAIT_WORLD[0]) > 0.01 or abs(uy - UAV_WAIT_WORLD[1]) > 0.01:
        sys.exit(f"UAV in {src} is at ({ux}, {uy}), agv_controller expects {UAV_WAIT_WORLD}")
    car = swept_world_poses(speed_kmh * KMH_TO_MS)
sdf = set_pose(sdf, "prius_hybrid", car)
open(dst, "w").write(sdf)
print(f"car ({car[0]:.3f}, {car[1]:.3f}, {car[2]:.1f} deg)")
PY
    )
}

wait_for_ekf() {    # $1 = run dir; MAVProxy and MAVROS both echo FCU status text
    local d="$1" t0=$SECONDS
    until grep -qs "is using GPS" "$d/sitl.log" "$d/mavros.log" \
          && grep -qs "Got HEARTBEAT" "$d/mavros.log"; do
        (( SECONDS - t0 > EKF_TIMEOUT )) && return 1
        sleep 1
    done
    sleep 2
}

watch_run() {   # $1 = start marker, $2 = launch pid -> prints "outcome,state,attempts,rtl_reason,csv"
    python3 - "$WS" "$1" "$2" "$RUN_TIMEOUT" "$TERMINAL_HOLD" <<'PY'
import csv, glob, os, sys, time

ws, marker, pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
timeout, hold = float(sys.argv[4]), float(sys.argv[5])
t_start, t_marker = time.time(), os.path.getmtime(marker)
path = header = last = since = None

def alive():
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def tail_row():
    """Last complete CSV row of the (still growing) controller log."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        start = max(0, f.tell() - 262144)
        f.seek(start)
        lines = f.read().decode(errors="ignore").split("\n")
    lines = (lines[1:] if start else lines)[:-1]   # drop seek-cut first line and partial last line
    for line in reversed(lines):
        row = next(csv.reader([line]), None)
        if row and len(row) == len(header):
            return dict(zip(header, row))
    return None

def report(outcome):
    g = (last or {}).get
    csv.writer(sys.stdout, lineterminator="").writerow(
        [outcome, g("controller_state", ""), g("landing_attempts", ""), g("rtl_reason", ""), path or ""])
    sys.exit(0)

while True:
    if time.time() - t_start > timeout:
        report("TIMEOUT")
    if not alive():
        report("LAUNCH_DIED")
    if path is None:
        found = [p for p in glob.glob(os.path.join(ws, "controller_*.csv"))
                 if os.path.getmtime(p) >= t_marker]
        if found:
            path = max(found, key=os.path.getmtime)
            with open(path, newline="") as f:
                header = next(csv.reader(f), None)
            if not header:          # header still sitting in the write buffer
                path = None
    else:
        rec = tail_row()
        try:
            state = int(float(rec["controller_state"]))
        except (TypeError, KeyError, ValueError):
            state = None
        if state is not None:
            last = rec
            if state >= 6000:
                since = since or time.time()
                if time.time() - since >= hold:
                    report("LANDED" if state < 7000 else "RTL")
            else:
                since = None
    time.sleep(1.0)
PY
}

collect() {     # $1 = run dir: move everything this run wrote into the workspace root
    find "$WS" -mindepth 1 -maxdepth 1 -newer "$1/.start" \
        ! -name build ! -name install ! -name log ! -name src ! -name bags \
        -exec mv -t "$1" {} + 2>/dev/null
}

run_one() {     # $1 = trajectory, $2 = speed (km/h), $3 = repeat
    local traj=$1 spd=$2 rep=$3 d attempt result launch_pid t0 spawn overrides
    local -a extra
    d=$(printf '%s/%s/%02dkmh/rep%02d' "$RESULTS" "$traj" "$spd" "$rep")
    [[ -f "$d/DONE" ]] && return

    t0=$SECONDS
    (( RUN_IDX++ ))
    phase=$(awk -v r="$rep" -v n="$SIN_PHASE_STRATA" 'BEGIN { printf "%.1f", (n > 0 ? ((r - 1) % n) * 360 / n : 0) }')
    for (( attempt = 1; attempt <= MAX_ATTEMPTS; attempt++ )); do
        rm -rf "$d"; mkdir -p "$d"
        RUN_WORLD="$WORLD"; spawn=""
        # Per-run world: mbzirc spawns the car at this repeat's lap phase; moving swept runs
        # spawn it far enough back to be at cruise before it comes into view. 0 km/h keeps
        # WORLD untouched (the static runs and their spawn are unchanged).
        if [[ $traj == mbzirc || $spd != 0 ]]; then
            if ! spawn=$(make_world "$d" "$traj" "$spd" "$phase" 2>"$d/world_gen.log"); then
                log "    could not write the run world (see $d/world_gen.log), stopping"
                exit 1
            fi
            RUN_WORLD="$d/world.sdf"
        fi
        log "[$RUN_IDX/$TOTAL] $traj  ${spd} km/h  rep $rep$([[ $traj == sin || $traj == mbzirc ]] && echo "  phase ${phase} deg")${spawn:+  $spawn}  (attempt $attempt)"
        if [[ $traj == mbzirc ]]; then      # null = launch-file default
            overrides="\"agv_delay_s\": $MBZIRC_AGV_DELAY_S, \"gps_lat\": $MBZIRC_GPS_LAT, \"gps_lon\": $MBZIRC_GPS_LON"
        else
            overrides='"agv_delay_s": null, "gps_lat": null, "gps_lon": null'
        fi
        printf '{"study": "%s", "trajectory": "%s", "speed_kmh": %s, "repeat": %s, "world": "%s", "lead_in": "%s", "car_spawn": "%s", "accel_distance_m": %s, "turn_start_distance_m": %s, "sin_phase_deg": %s, %s}\n' \
            "$STUDY" "$traj" "$spd" "$rep" "$RUN_WORLD" \
            "$([[ $traj == mbzirc ]] && echo "mbzirc_ramp" || echo "prius_scurve_cruise_line")" "$spawn" \
            "$ACCEL_DISTANCE_M" "$TURN_START_DISTANCE_M" "$phase" \
            "$overrides" > "$d/run_info.json"

        start_sim "$d"
        if ! wait_for_ekf "$d"; then
            result="NO_EKF,,,,"
            log "    EKF/MAVROS not ready after ${EKF_TIMEOUT}s, restarting sim"
            kill_sim; continue
        fi

        touch "$d/.start"
        extra=()                            # mbzirc-only launch overrides
        [[ $traj == mbzirc ]] && extra=(agv_delay_s:="$MBZIRC_AGV_DELAY_S"
                                        gps_lat:="$MBZIRC_GPS_LAT" gps_lon:="$MBZIRC_GPS_LON")
        (
            ros_env; cd "$WS"
            exec ros2 launch auto_lander main_launch_sim.py \
                trajectory:="$traj" speed_kmh:="${spd}.0" \
                accel_distance_m:="$ACCEL_DISTANCE_M" turn_start_distance_m:="$TURN_START_DISTANCE_M" \
                sin_phase_deg:="$phase" "${extra[@]}"
        ) </dev/null >"$d/launch.log" 2>&1 &
        launch_pid=$!

        result=$(watch_run "$d/.start" "$launch_pid")
        kill -INT "$launch_pid" 2>/dev/null
        wait_pid "$launch_pid" 30           # let the controller close its CSV
        kill_sim
        collect "$d"

        case "$result" in
            TIMEOUT*|LAUNCH_DIED*) log "    ${result%%,*}, retrying"; continue ;;
        esac
        break
    done

    (( attempt > MAX_ATTEMPTS )) && attempt=$MAX_ATTEMPTS
    echo "$traj,$spd,$rep,$result,$attempt,$((SECONDS - t0)),$d" >> "$SUMMARY"
    touch "$d/DONE"
    log "    -> ${result%%,*}  ($((SECONDS - t0)) s)"
}

# =============================================================================
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.csv"
[[ -f "$SUMMARY" ]] || echo "trajectory,speed_kmh,repeat,outcome,final_state,landing_attempts,rtl_reason,controller_csv,attempts,wall_s,run_dir" > "$SUMMARY"

trap 'echo; log "Interrupted, tearing down"; kill_sim; exit 130' INT TERM

log "Building auto_lander"
( source /opt/ros/humble/setup.bash && cd "$WS" && colcon build --packages-select auto_lander ) \
    >>"$RESULTS/build.log" 2>&1 || { log "Build failed, see $RESULTS/build.log"; exit 1; }

# Count the runs (for progress only)
TOTAL=0; RUN_IDX=0
for traj in "${TRAJECTORIES[@]}"; do for spd in "${SPEEDS_KMH[@]}"; do
    [[ $ZERO_SPEED_ONCE == 1 && $spd == 0 && $traj != "${TRAJECTORIES[0]}" ]] && continue
    (( TOTAL += REPEATS ))
done; done
log "Study '$STUDY': $TOTAL runs, world $WORLD, speeds [${SPEEDS_KMH[*]}] km/h, trajectories [${TRAJECTORIES[*]}]"
log "    -> $RESULTS (Prius S-curve lead-in, turns ${TURN_START_DISTANCE_M} m past the cruise line; mbzirc ramp ${ACCEL_DISTANCE_M} m)"

kill_sim    # clear anything left over from a manual session

# Repeat is the outer loop, so stopping early still leaves a balanced dataset
for (( rep = 1; rep <= REPEATS; rep++ )); do
    for traj in "${TRAJECTORIES[@]}"; do
        for spd in "${SPEEDS_KMH[@]}"; do
            [[ $ZERO_SPEED_ONCE == 1 && $spd == 0 && $traj != "${TRAJECTORIES[0]}" ]] && continue
            [[ -f $(printf '%s/%s/%02dkmh/rep%02d/DONE' "$RESULTS" "$traj" "$spd" "$rep") ]] && { (( RUN_IDX++ )); continue; }
            run_one "$traj" "$spd" "$rep"
        done
    done
done

log "Study complete: $SUMMARY"