#!/usr/bin/env bash
# Ground-station start-up for this drone, driven by fpv_switch.py ("ctrl ...") from
# the console's controller panel. Same shape as the 15" drone's ugv_stack.sh.
#
#   goto_stack.sh prepare  MAVROS only (no controller): run it while still disarmed,
#                          so "start" after arming only has to launch the node.
#   goto_stack.sh start    MAVROS if not up, then goto_hold.py. The node never arms:
#                          the pilot arms with the transmitter, presses Start, and it
#                          sets GUIDED, takes off to $HEIGHT m, flies to
#                          $TARGET_LAT,$TARGET_LON, turns to $YAW and holds; RTL
#                          $FLIGHT_TIME s after take-off.
#   goto_stack.sh lander   hand over to the auto_lander stack (this repo's main_launch.py)
#                          once goto_hold.py is HOLDING at the target: the go-to node is
#                          stopped (the FCU keeps holding its last target in GUIDED), the
#                          camera publisher is started, then the lander launch with the
#                          go-to target as its GPS waiting point. Refused otherwise.
#   goto_stack.sh stop     stop the go-to node / the lander (drone stays in GUIDED - take
#                          over; after the lander it may NOT be holding position)
#   goto_stack.sh down     stop everything this script started
#   goto_stack.sh status   one line: "ctrl running|off stack mavros:on|off det:off mission:off | <last message>"
#
# Settings come from the environment (fpv_switch.py fills them from the console):
# TARGET_LAT TARGET_LON HEIGHT YAW SPEED FLIGHT_TIME ARMED_EPOCH.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$HERE/logs"; mkdir -p "$LOGDIR"
MSG=/tmp/auto_stack.msg
FCU_URL="${FCU_URL:-tcp://127.0.0.1:5760}"   # wfb-ng's local MAVLink proxy (it owns the FC UART)
TARGET_LAT="${TARGET_LAT:-0.0}"
TARGET_LON="${TARGET_LON:-0.0}"
HEIGHT="${HEIGHT:-5.0}"                      # m above where it is armed
YAW="${YAW:--1.0}"                           # compass deg; negative = keep the heading at arming
SPEED="${SPEED:-1.0}"                        # m/s on the way to the target
FLIGHT_TIME="${FLIGHT_TIME:-120.0}"          # s after take-off until RTL
ARM_SETTLE="${ARM_SETTLE:-3.0}"              # s after arming before the take-off request
MAX_DISTANCE="${MAX_DISTANCE:-80.0}"         # m: refuse a target further than this
LANDER_DIR="$(dirname "$HERE")"              # the DroneAutoLander clone (cwd of the launch: bags/, model path)
LANDER_WS="${LANDER_WS:-$HOME/lander_ws}"    # colcon workspace with auto_lander built
LANDER_TIME="${LANDER_TIME:-180.0}"          # s: the lander's max_runtime (it RTLs after this)
LANDER_BOUNDARY="${LANDER_BOUNDARY:-300.0}"  # m: the lander RTLs if |x| or |y| from home exceeds this (its default is 80)
RTSP_URL="${RTSP_URL:-rtsp://192.168.144.25:8554/main.264}"
# Camera model: the A2's fisheye picture is remapped to a pinhole by rtsp_camera.py.
# FISHEYE_F is the lens scale in px/rad at 640 px wide (about 230 measured in flight;
# 215 errs towards "the car is closer than it is"). CAM_HFOV must equal
# CAMERA_HFOV_RAD in the lander's main_launch.py (120 deg).
FISHEYE_F="${FISHEYE_F:-215.0}"
CAM_HFOV="${CAM_HFOV:-120.0}"
CAM_LATENCY="${CAM_LATENCY:-0.2}"            # s: camera capture-to-read delay, taken off each image stamp
YOLO_HEF="${YOLO_HEF:-$HOME/hailo/models/car_train4.hef}"   # car detector on the Hailo; missing file = CPU model

say() { printf '[goto_stack %s] %s\n' "$(date +%T)" "$*"; printf '%s\n' "$*" > "$MSG"; }
mavros_pid() { pgrep -f "[m]avros_node" | head -1; }
goto_pid()   { pgrep -f "[g]oto_hold.py" | head -1; }
lander_pid() { pgrep -f "[l]ib/auto_lander/controller" | head -1; }
launch_pid() { pgrep -f "[r]os2 launch auto_lander main_launch.py" | head -1; }
cam_pid()    { pgrep -f "[r]tsp_camera.py" | head -1; }
gimbal_pid() { pgrep -f "[g]imbal_bridge.py" | head -1; }
ctrl_pid()   { goto_pid; lander_pid; }

ros_env() {
    export ROS_LOCALHOST_ONLY=1      # everything is on this Pi; keep other machines' MAVROS out
    source /opt/ros/jazzy/setup.bash
    cd "$HERE" || exit 1
}

set_rate() {   # set_rate <msgid> <hz>
    timeout 10 ros2 service call /mavros/set_message_interval \
        mavros_msgs/srv/MessageInterval "{message_id: $1, message_rate: $2}" >/dev/null 2>&1
}

start_mavros() {
    if [ -n "$(mavros_pid)" ]; then say "mavros already running"; return 0; fi
    say "starting MAVROS ($FCU_URL)"
    # Stock MAVROS config with one change: time-sync accepts round trips up to 50 ms.
    # Through wfb-ng's proxy on a 115200 baud UART the round trip is 11-22 ms, over the
    # stock 10 ms limit, so every sample was being rejected ("RTT too high for timesync").
    sed 's/^\( *\)timesync_mode: MAVLINK.*/&\n\1max_rtt_sample: 50/' \
        /opt/ros/jazzy/share/mavros/launch/apm_config.yaml > "$HERE/apm_config_goto.yaml"
    grep -q "max_rtt_sample: 50" "$HERE/apm_config_goto.yaml" || { say "ERROR: could not build the MAVROS config"; return 1; }
    : > "$LOGDIR/mavros.log"
    setsid nohup ros2 launch "$HERE/mavros_goto.launch" fcu_url:="$FCU_URL" \
        pluginlists:="$HERE/mavros_pluginlists.yaml" config:="$HERE/apm_config_goto.yaml" \
        </dev/null >"$LOGDIR/mavros.log" 2>&1 &
    for i in $(seq 1 45); do
        grep -q "Got HEARTBEAT" "$LOGDIR/mavros.log" 2>/dev/null && break
        sleep 1
    done
    grep -q "Got HEARTBEAT" "$LOGDIR/mavros.log" 2>/dev/null || { say "ERROR: no FCU heartbeat through MAVROS (wfb-ng proxy on 5760 up?)"; return 1; }
    say "FCU heartbeat via MAVROS; setting stream rates"
    #   32 LOCAL_POSITION_NED 10 Hz, 33 GLOBAL_POSITION_INT 5 Hz, 74 VFR_HUD (compass) 5 Hz
    set_rate 32 10.0; set_rate 33 5.0; set_rate 74 5.0
    sleep 2
    POSE=$(timeout 8 ros2 topic echo /mavros/local_position/pose --once 2>/dev/null | grep -c "position:")
    FIX=$(timeout 8 ros2 topic echo /mavros/global_position/global --once 2>/dev/null | grep -m1 "latitude:" | awk '{print $2}')
    [ "$POSE" = 1 ] || { say "ERROR: no local position from the FCU (EKF origin not set yet? wait for GPS)"; return 1; }
    [ -n "$FIX" ] || { say "ERROR: no GPS position from the FCU"; return 1; }
}

prepare() {
    ros_env
    start_mavros || return 1
    say "prepared (go to position): arm with the transmitter, then Start"
}

start() {
    ros_env
    start_mavros || return 1
    if [ -n "$(ctrl_pid | head -1)" ]; then say "controller already running"; return 0; fi
    LOG="$LOGDIR/goto_$(date +%Y%m%d_%H%M%S).log"
    say "starting go-to: takeoff ${HEIGHT} m -> ${TARGET_LAT},${TARGET_LON} heading ${YAW} at ${SPEED} m/s -> RTL after ${FLIGHT_TIME} s"
    setsid nohup python3 "$HERE/goto_hold.py" --ros-args \
        -p target_lat:="$(printf "%.7f" "$TARGET_LAT")" -p target_lon:="$(printf "%.7f" "$TARGET_LON")" \
        -p height:="$(printf "%.2f" "$HEIGHT")" -p yaw_deg:="$(printf "%.1f" "$YAW")" \
        -p speed:="$(printf "%.2f" "$SPEED")" -p max_flight_time:="$(printf "%.1f" "$FLIGHT_TIME")" \
        -p max_distance:="$(printf "%.1f" "$MAX_DISTANCE")" -p arm_settle_s:="$(printf "%.1f" "$ARM_SETTLE")" \
        -p armed_epoch:="$(printf "%.2f" "${ARMED_EPOCH:-0.0}")" \
        </dev/null >>"$LOG" 2>&1 &
    sleep 3
    if [ -n "$(goto_pid)" ]; then say "controller running (pid $(goto_pid)) - go to position"; else say "ERROR: controller exited, see $LOG"; return 1; fi
}

lander() {
    ros_env
    [ -n "$(lander_pid)" ] && { say "lander already running"; return 0; }
    [ -n "$(mavros_pid)" ] || { say "ERROR: MAVROS is not running - lander not started"; return 1; }
    [ -n "$(goto_pid)" ] || { say "ERROR: the go-to controller is not running - lander not started"; return 1; }
    grep -q "^goto: holding at target" "$MSG" 2>/dev/null || { say "ERROR: not holding at the target yet - lander not started ($(cat "$MSG" 2>/dev/null))"; return 1; }
    [ -f "$LANDER_WS/install/setup.bash" ] || { say "ERROR: $LANDER_WS is not built - lander not started"; return 1; }
    [ "$TARGET_LAT" != "0.0" ] || { say "ERROR: no waiting point - lander not started"; return 1; }
    source "$LANDER_WS/install/setup.bash"
    if [ -z "$(cam_pid)" ]; then
        setsid nohup python3 "$HERE/rtsp_camera.py" --ros-args -p url:="$RTSP_URL" \
            -p fisheye_f:="$FISHEYE_F" -p out_hfov_deg:="$CAM_HFOV" -p latency_s:="$CAM_LATENCY" \
            </dev/null >"$LOGDIR/rtsp_camera.log" 2>&1 &
    fi
    # The lander's gimbal commands reach the A2 mini through this bridge (the FC has no
    # mount on this drone); it also publishes the gimbal angle the lander reads.
    if [ -z "$(gimbal_pid)" ]; then
        setsid nohup python3 "$HERE/gimbal_bridge.py" </dev/null >"$LOGDIR/gimbal_bridge.log" 2>&1 &
    fi
    # Only now let go: the FCU keeps flying to / holding its last position target.
    P=$(goto_pid); kill "$P" 2>/dev/null; sleep 0.5; kill -9 "$P" 2>/dev/null
    LOG="$LOGDIR/lander_$(date +%Y%m%d_%H%M%S).log"
    say "starting the lander: waiting point ${TARGET_LAT},${TARGET_LON}, RTL after ${LANDER_TIME} s or beyond ${LANDER_BOUNDARY} m"
    cd "$LANDER_DIR" || return 1
    setsid nohup ros2 launch auto_lander main_launch.py \
        gps_lat:="$(printf "%.7f" "$TARGET_LAT")" gps_lon:="$(printf "%.7f" "$TARGET_LON")" \
        max_runtime:="$(printf "%.1f" "$LANDER_TIME")" \
        boundary_limit:="$(printf "%.1f" "$LANDER_BOUNDARY")" \
        yolo_model:="$LANDER_DIR/ugv_yolo11n_openvino_model" \
        yolo_hef:="$([ -f "$YOLO_HEF" ] && echo "$YOLO_HEF")" \
        </dev/null >>"$LOG" 2>&1 &
    for i in $(seq 1 15); do [ -n "$(lander_pid)" ] && break; sleep 1; done
    if [ -n "$(lander_pid)" ]; then say "lander running (pid $(lander_pid)) - Stop or the transmitter takes it back"
    else say "ERROR: lander did not start, see $LOG - drone is holding in GUIDED, take over"; return 1; fi
}

bag_pids()   { pgrep -f "[r]os2 bag record .*DroneAutoLander/bags/"; }

stop() {
    L=$(launch_pid); G=$(goto_pid)
    if [ -n "$L" ] || [ -n "$(lander_pid)" ] || [ -n "$(bag_pids)" ]; then
        # The controller must stop commanding at once; the bag needs a clean SIGINT and a
        # few seconds or its file is unreadable (every bag before 2026-11-13 14:40 was lost
        # that way, and one recorder was left running after the launch had gone).
        # SIGTERM, not SIGINT: everything here is started in the background from a script,
        # which leaves SIGINT ignored in the bag recorder (found 2026-11-13).
        for P in $(pgrep -f "[l]ib/auto_lander/controller"); do kill -INT "$P" 2>/dev/null; kill -TERM "$P" 2>/dev/null; done
        [ -n "$L" ] && kill -TERM "$L" 2>/dev/null
        for i in 1 2 3 4 5 6; do [ -z "$(lander_pid)" ] && break; sleep 0.5; done
        pkill -9 -f "[l]ib/auto_lander/" 2>/dev/null     # the launch's nodes only
        say "lander stopped - TAKE OVER on the transmitter now (GUIDED without setpoints does not hold position)"
        for P in $(bag_pids); do kill -TERM "$P" 2>/dev/null; done
        for i in $(seq 1 20); do [ -z "$(bag_pids)" ] && break; sleep 0.5; done
        for P in $(bag_pids); do kill -9 "$P" 2>/dev/null; done
        P=$(cam_pid); [ -n "$P" ] && kill "$P" 2>/dev/null
        P=$(gimbal_pid); [ -n "$P" ] && { kill -INT "$P" 2>/dev/null; kill -TERM "$P" 2>/dev/null; }
    elif [ -n "$G" ]; then kill "$G" 2>/dev/null; sleep 1; kill -9 "$G" 2>/dev/null; say "controller stopped (drone holds in GUIDED - take over on the transmitter)"
    else say "controller not running"; fi
}

down() {
    stop
    P=$(cam_pid); [ -n "$P" ] && kill "$P" 2>/dev/null
    P=$(gimbal_pid); [ -n "$P" ] && { kill -INT "$P" 2>/dev/null; kill -TERM "$P" 2>/dev/null; }
    pkill -f "[m]avros_node"; pkill -f "[m]avros_goto.launch"
    sleep 1; say "stack down"
}

status() {
    C=$(ctrl_pid | head -1); M=$(mavros_pid)
    printf 'ctrl %s stack mavros:%s det:off mission:off | %s\n' \
        "$([ -n "$C" ] && echo running || echo off)" "$([ -n "$M" ] && echo on || echo off)" \
        "$(cat "$MSG" 2>/dev/null)"
}

case "$1" in
    prepare) prepare ;;
    start)   start ;;
    lander)  lander ;;
    stop)    stop ;;
    down)    down ;;
    status)  status ;;
    *) echo "usage: $0 prepare|start|lander|stop|down|status"; exit 1 ;;
esac
