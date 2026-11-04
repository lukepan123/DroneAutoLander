#!/usr/bin/env bash
# =============================================================================
#  run_sweep.sh : static YOLO box-fit sweep, using the same sim startup as run_batch.sh
#
#    1. start Gazebo, ArduPilot SITL and MAVROS exactly as run_batch.sh does
#    2. wait for "EKF3 IMU0 is using GPS" and a MAVROS heartbeat
#    3. ros2 launch auto_lander with AUTO_LANDER_SWEEP=1 (controller logs, never flies)
#    4. sweep_runner.py flies the drone and teleports the car through every case
#    5. shut down cleanly, move the controller CSV into the run folder, analyse
#
#  Usage   : ./run_sweep.sh               full sweep
#            ./run_sweep.sh --limit 3     smoke test (extra args go to sweep_runner.py)
#  Resume  : run it again with the same SWEEP_TAG; completed cases are skipped
#            (delete sweep_out/<tag>/manifest.csv to restart that condition)
#  Stop    : Ctrl+C (the sim is torn down on exit)
#
#  Conditions (SWEEP-TEMP), all optional, inherited by every launched node:
#    SWEEP_TAG=name             output folder sweep_out/<name> (default: "default")
#    AUTO_LANDER_RAYCAST=1      ray-cast baseline instead of the projected mesh
#    AUTO_LANDER_ROOF_INIT=3.0  assumed pad height (m above home); unset = true height
#    AUTO_LANDER_TRUE_YAW=1     hand the true car heading to the fit (both methods)
#  e.g. SWEEP_TAG=raycast_wrong AUTO_LANDER_RAYCAST=1 AUTO_LANDER_ROOF_INIT=3.0 ./run_sweep.sh
#  run_conditions.sh runs all four in sequence.
# =============================================================================
set -o pipefail
set -m                      # job control: background jobs keep SIGINT, so launch can shut down cleanly

# ---------------- Paths (same as run_batch.sh) ----------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this folder: scripts + sweep_out
WS="$HOME/ros2_ws"
ARDUPILOT="$HOME/ardupilot"
ARDU_GZ="$HOME/ardupilot_gazebo"
WORLD=${WORLD:-iris_runway_new.sdf}   # file in ~/ardupilot_gazebo/worlds; keep <world name="iris_runway_new">

# SWEEP-TEMP: one output folder per condition, shared with the Python scripts
SWEEP_TAG=${SWEEP_TAG:-default}
OUT="$HERE/sweep_out/$SWEEP_TAG"
export SWEEP_OUT="$OUT"

# ---------------- Timing (s) ----------------
EKF_TIMEOUT=180

# ---------------- Display ----------------
SHOW_GUI=${SHOW_GUI:-0}     # 0 = Gazebo server only + no MAVProxy console/map; SHOW_GUI=1 to watch

if (( SHOW_GUI )); then
    GZ_FLAGS="";                        SITL_UI="--console --map"
else
    GZ_FLAGS="-s --headless-rendering"; SITL_UI=""
fi

export PYTHONUNBUFFERED=1
export RCUTILS_LOGGING_BUFFERED_STREAM=0

# =============================================================================
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
        cd "$ARDU_GZ/worlds" && exec gz sim $GZ_FLAGS "$WORLD" -v -r
    ) </dev/null >"$d/gazebo.log" 2>&1 &
    sleep 5

    # Same as run_batch.sh plus a second output (14556) for sweep_runner.py
    (
        cd "$ARDUPILOT" && exec "$ARDUPILOT/Tools/autotest/sim_vehicle.py" -v ArduCopter $SITL_UI \
            -f JSON \
            --add-param-file="$ARDU_GZ/config/gazebo-iris-gimbal_1d.parm" \
            --out=udp:127.0.0.1:14555 \
            --out=udp:127.0.0.1:14556 \
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

wait_for_ekf() {    # $1 = run dir; MAVProxy and MAVROS both echo FCU status text
    local d="$1" t0=$SECONDS
    until grep -qs "is using GPS" "$d/sitl.log" "$d/mavros.log" \
          && grep -qs "Got HEARTBEAT" "$d/mavros.log"; do
        (( SECONDS - t0 > EKF_TIMEOUT )) && return 1
        sleep 1
    done
    sleep 2
}

collect() {     # $1 = run dir: move everything this run wrote into the workspace root
    find "$WS" -mindepth 1 -maxdepth 1 -newer "$1/.start" \
        ! -name build ! -name install ! -name log ! -name src ! -name bags ! -name "$(basename "$HERE")" \
        -exec mv -t "$1" {} + 2>/dev/null
}

# =============================================================================
mkdir -p "$OUT"
RUN_DIR="$OUT/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "Condition: $SWEEP_TAG  (RAYCAST=${AUTO_LANDER_RAYCAST:-0}, ROOF_INIT=${AUTO_LANDER_ROOF_INIT:-true height}, TRUE_YAW=${AUTO_LANDER_TRUE_YAW:-0})"
echo "Logs -> $RUN_DIR"

# SWEEP-TEMP: record the condition next to the data, so a folder is never ambiguous
{
    echo "SWEEP_TAG=$SWEEP_TAG"
    echo "AUTO_LANDER_RAYCAST=${AUTO_LANDER_RAYCAST:-0}"
    echo "AUTO_LANDER_ROOF_INIT=${AUTO_LANDER_ROOF_INIT:-unset}"
    echo "AUTO_LANDER_TRUE_YAW=${AUTO_LANDER_TRUE_YAW:-0}"
    echo "started=$(date -Iseconds)"
} >"$RUN_DIR/condition.txt"

trap 'echo; echo "Interrupted, tearing down"; kill_sim; exit 130' INT TERM

echo "Building auto_lander"
( source /opt/ros/humble/setup.bash && cd "$WS" && colcon build --packages-select auto_lander ) \
    >>"$RUN_DIR/build.log" 2>&1 || { echo "Build failed, see $RUN_DIR/build.log"; exit 1; }

kill_sim    # clear anything left over from a manual session

echo "Starting sim"
start_sim "$RUN_DIR"
if ! wait_for_ekf "$RUN_DIR"; then
    echo "EKF/MAVROS not ready after ${EKF_TIMEOUT}s, see $RUN_DIR/sitl.log and mavros.log"
    kill_sim; exit 1
fi
echo "EKF on GPS, MAVROS connected"

touch "$RUN_DIR/.start"
(
    ros_env; cd "$WS"
    export AUTO_LANDER_SWEEP=1
    exec ros2 launch auto_lander main_launch_sim.py
) </dev/null >"$RUN_DIR/launch.log" 2>&1 &
launch_pid=$!
sleep 10                    # bridge, YOLO model load, gimbal

( ros_env; cd "$HERE" && python3 sweep_runner.py --run-dir "$RUN_DIR" "$@" )

# Stop YOLO on its own first. On Ctrl+C it stitches its saved frames into the video, which
# takes minutes after a long sweep; stopping it through the launch would cut that short
# (20 s SIGTERM timeout, and a second Ctrl+C makes it force-quit without a video).
yolo_pid=$(pgrep -f "lib/auto_lander/yolo" | head -n 1)
if [[ -n $yolo_pid ]]; then
    echo "Waiting for YOLO to write its video (can take a few minutes)..."
    kill -INT "$yolo_pid" 2>/dev/null
    wait_pid "$yolo_pid" 900
fi

kill -INT "$launch_pid" 2>/dev/null
wait_pid "$launch_pid" 30   # let the controller close its CSV
kill_sim
collect "$RUN_DIR"

# SWEEP-TEMP: sanity check that the condition actually reached the controller
grep -h "SWEEP-TEMP" "$RUN_DIR/launch.log" | sort | uniq -c | head -n 5

( cd "$HERE" && python3 analyse_sweep.py && python3 plot_sweep.py )
