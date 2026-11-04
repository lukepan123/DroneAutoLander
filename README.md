# DroneAutoLander

## Resources
https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html

## Running IRL

All real-flight settings live in `main_launch.py`. Items marked `TODO` at the top of that file are SITL placeholders that **must** be replaced with measured values before flying.

### Hardware

**Conflicts with this stack**
- **No camera publisher.** The A2 mini is RTSP, not a USB webcam, so the AprilTag node's webcam mode can't open it and nothing publishes `/camera/image_raw` yet.
- **Tilt control path and angle frame.** The camera pose code assumes the reported gimbal pitch is relative to the drone body, which is how the SITL servo gimbal behaves. The A2 stabilises itself, so on the real drone the reported pitch is likely earth-referenced (bench GUI notes: the SDK readback is gravity-referenced), and body pitch would be counted twice. Bench test: pitch the drone 20 deg nose-down with the gimbal holding; if `/landing_pad/gimbal_angle` stays constant, the pose code needs changing before flight.
- **57600 baud is too slow** for 20 Hz-plus odometry plus 20 Hz setpoints. Raise `SERIAL6_BAUD` to 921 and update the 57600 in `fpv_switch.py` and `mnt_center.py` to match.

### Pre-flight checklist

**Before the field day (config)**
- [ ] Camera calibrated at the processed resolution (16:9): `CAMERA_HFOV_RAD`, `CAMERA_DISTORTION`
- [ ] Camera mount measured (translation + rotation): `CAMERA_MOUNT`
- [ ] Car and pad measured: `CAR_BOXES`, `PAD_HEIGHT_M`, `CAR_PAD_POSITION`, `CAR_REAR_AXLE`, `PAD_SIZE`
- [ ] FCU parameters set (see [FCU parameters](#fcu-parameters)), geofence on
- [ ] GPS Waiting point chosen, within `boundary_limit` of home

**At the field (bench, props off)**
- [ ] `ros2 param get /mavros/setpoint_raw thrust_scaling` returns 1.0 (otherwise ArduPilot ignores every attitude setpoint)
- [ ] MAVROS on the serial link, **without** `use_sim_time`
- [ ] `ros2 topic hz /mavros/global_position/local` reads **at least 20 Hz**
- [ ] `/camera/image_raw` is publishing and YOLO boxes the car
- [ ] Gimbal moves, and `/landing_pad/gimbal_angle` follows the real angle
- [ ] No `UKF loop took too long` / `Control loop took too long` warnings, CPU temperature OK

### Launch

```bash
# MAVROS on the companion UART (after raising SERIAL6_BAUD to 921)
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=serial:///dev/ttyAMA0:921600 \
  --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_config.yaml \
  --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_pluginlists.yaml

# Auto lander (the launch refuses to start without a waiting point)
ros2 launch auto_lander main_launch.py gps_lat:=<deg> gps_lon:=<deg>
```

| Launch argument | Default | Notes |
|---|---|---|
| `gps_lat`, `gps_lon` | required | Waiting point. Refused (RTL) if further than `boundary_limit` from the drone |
| `boundary_limit` | 80 m | RTL if \|x\| or \|y\| from home exceeds this |
| `max_runtime` | 180 s | RTL this long after takeoff. Check against battery endurance |
| `use_webcam` | false | true only for a USB/V4L2 webcam. The A8 mini needs an RTSP publisher for `/camera/image_raw` |
| `yolo_model` | `ugv_sim_yolo11n_openvino_model` | Absolute path, or relative to the cwd |
| `record_bag` | true | Key topics, no images |

### How a real run behaves
- **Start:** Goes straight into GUIDED, waits 5 secs, and takes off to 8.5 m (chase altitude + pad height), flies to the waiting point, searches, tracks and lands. (An option exists to manually fly to the point and then switch to GUIDED)
- **Pilot override:** leaving GUIDED after arming makes the controller passive for the rest of the run (state 8000): no setpoints, no RTL, no disarm. Relaunch to retry.
- **Landing:** throttle cut at 1 m or below once the termination check passes, then a **force disarm 0.5 s later**.
- **Safety RTL:** boundary, `max_runtime`, UKF covariance blow-up, more than 3 failed attempts, or a waiting point beyond `boundary_limit`. Only ever issued while in GUIDED.
- **Touchdown fallback** (big accel_z detection cut) is **disabled** for first flights (`landing_z_touchdown_accel_threshold: 1000.0`). `quad_z_accel` is logged so it can be tuned.

**Known limitations (pending fix)**
- It always flies to the waiting point; there is no option to search from where the pilot hands over.
- If the pilot flies manually above 8 m before selecting GUIDED, the `max_runtime` timer starts early, so GUIDED can trigger an immediate RTL after more than 180 s of manual flight.
- Avoid switching to guided below 2 m.

### What to check, by component

**Camera** (`CAMERA_*` constants, shared by the YOLO and AprilTag nodes)
1. `CAMERA_HFOV_RAD` is the A2 mini's 160 deg spec, which is also the SITL camera. The A2 corrects distortion in-camera, so the real question is how close its output is to the ideal pinhole SITL assumes: calibrate with `ros2 run auto_lander camera_calibrate` on the 1080p stream and check the residual distortion, especially near the image edges. If it is large, the pipeline needs a fisheye model, which it doesn't support yet.
2. Frames are stamped when the Pi reads them, but RTSP adds encode, transport and decode delay (likely 100-200 ms). Measure it once (film a running stopwatch, compare) because it is latency the UKF's OOSM handling cannot see.
3. `CAMERA_MOUNT` is read by the perception nodes, not the gimbal controller. Offsets are from the drone body origin to the tilt axis.
4. The stream is H.265 by default; decoding it on the Pi costs more CPU than H.264, which competes with YOLO. Switch the gimbal to H.264 if it supports it.

**YOLO**
1. The model was trained on sim imagery. Fine-tune on real car footage, then recheck `yolo_conf_threshold` (0.60) and `yolo_car_class_id` (0). If you fall back to the stock COCO `yolo11n` already on the Pi (`/home/case/yolo11n_openvino_model`), car is **class 2**, not 0.
2. OpenVINO with a fixed 384x640 export already runs on the Pi (`pi_yolo_stream.py`), so the format is fine. Measure the real rate (`yolo_inference_lag` and `yolo_total_lag` in the CSV); the camera caps it at 25 fps anyway. Use an absolute model path on the Pi.
3. Car shape and pad height live on the controller. `PAD_HEIGHT_M` feeds both `landing_height_above_gnd` and the pad position, so they can't drift apart.

**Gimbal**
1. The camera pose uses the angle from `/mavros/gimbal_control/device/attitude_status` and assumes it is **relative to the drone body**; see the bench test under Conflicts. Also confirm the topic actually publishes (MAVROS `gimbal_control` plugin enabled).
2. How ArduPilot drives the tilt decides both the mount type and what that angle means:
   - **PWM** (the SITL setup, `SERVO10_FUNCTION 7`): use `MNT1_TYPE 7` (Brushless PWM), not 1. Type 1 adds ArduPilot's own stabilisation on top of the A2's, which is right for the unstabilised SITL joint but double-stabilises the real gimbal. With PWM there is no position feedback: the reported angle is ArduPilot's own target.
   - **SIYI serial** (`MNT1_TYPE 8`): real feedback, but the bench GUI found the A2's angle command inverted (and its soft limits on the wrong side) when the gimbal detects an upside-down mount. Check the tilt direction and both stops through the real command path before flying.
3. Limits: spec -90 / +25 deg, but the bench measured the lower hard stop at about -85. `gimbal_servo_min_angle_deg` -90 still works (the command clips), but -85 avoids pushing into the stop.
4. Gains (`gimbal_kp` 0.03, `gimbal_kd` 0.005, slew 60 deg/s) were tuned on an ideal sim servo. Start low and watch for hunting or jitter.

**Controller**
1. Landing geometry and timers mirror SITL (chase 7 m, recovery 7 m, descent -1.0 m/s, cut at 1 m or below). For first landings consider `landing_descent_rate_far/close` of -0.5.
2. Set the ArduPilot geofence as well as `boundary_limit`; check `RTL_ALT` clears obstacles at the site.

**UKF**
1. Pixel-box noise is exposed (`UKF_yolo_box_pixel_sigma`, 5 px). The fallback-fit noise and close-range inflation are still hardcoded in `controller.py::_UKF_loop`. Expect real images to need more noise than SITL.
2. `UKF_buffer_window_s` is 1.0 s for Pi latency. Drop `UKF_freq` from 40 to 20 if the loop overruns.

**PID / vehicle**
1. `PID_mass` is 1.5 kg (all-up weight). Throttle is handled by the FCU (thrust field = climb rate), so there is no thrust curve to match.
2. `PID_vz_up_max` / `PID_vz_dn_max` (2.0) must equal `WPNAV_SPEED_UP` / `WPNAV_SPEED_DN` / 100, and `PID_max_tilt` (0.70 rad) needs `ANGLE_MAX` of at least 4000.
3. Guidance gains mirror SITL (the SITL airframe is tuned to this quad). If the response looks aggressive, scale `PID_Kp_0`, `PID_Kd_0` and `PID_lam_0` down together. Test terminal shaping (`PID_terminal_gain`, `PID_drop_off*`, `PID_d_*`) last and from altitude.

### FCU parameters
SITL loads `gazebo-iris-gimbal_1d.parm` (1-axis PWM servo gimbal); none of it exists on the real FCU unless set.

| Parameter | Value | Notes |
|---|---|---|
| `FRAME_CLASS` / `FRAME_TYPE` | 1 / 1 | Quad X |
| `MOT_PWM_MIN` / `MOT_PWM_MAX` | 1100 / 1900 | Match your ESCs |
| `MNT1_TYPE` | 7 (PWM) or 8 (SIYI serial) | **Not 1** on the real A2: see Gimbal item 2 |
| `MNT1_DEFLT_MODE` | 2 | MAVLink targeting |
| `MNT1_PITCH_MIN` / `MNT1_PITCH_MAX` | -90 / 25 | |
| `SERVO10_FUNCTION`, `SERVO10_MIN/MAX` | 7, 1100 / 1900 | PWM control only; match the A2's PWM range |
| `RC7_OPTION`, `RC7_MIN/MAX` | 213, 1100 / 1900 | As SITL, manual tilt from the RC |
| `BRD_SERx_RTSCTS` | 0 | SIYI serial only, on a 3-wire cable |
| `SERIAL6_PROTOCOL` / `SERIAL6_BAUD` | 2 / **921** | Companion link, currently 57600 |
| `SR6_POSITION`, `SR6_EXTRA1` | 20 or more | Odometry + attitude to the Pi. Or set `mavros_message_rate_hz: 50.0` in the launch |
| `WPNAV_SPEED_UP` / `WPNAV_SPEED_DN` | 200 / 200 | Must match `PID_vz_*_max` |
| `ANGLE_MAX` | 4000 or more | Controller commands up to 40 deg |
| `GUID_OPTIONS` | bit 3 **clear** | Thrust read as climb rate, as in SITL |
| `FENCE_*` | site specific | Backup to `boundary_limit` |
| `RTL_ALT` | site specific | Default 15 m |
| Flight modes | GUIDED + LOITER/LAND on the mode switch | Pilot takeover |

Also check MAVROS timesync is healthy: image and odometry timestamps must share a clock for the camera pose interpolation.

**Legacy:** `landing_pad_detector.py` is the old combined node. Do not launch it; it drives the gimbal servo directly.


## Running in Simulator
**General Notes**

Each application will need to be run in separate terminal windows.

If after running `source install/setup.bash` no ROS2 commands are found, you will also need to run `source /opt/ros/humble/setup.bash`

To rebuild package:

    colcon build --packages-select auto_lander

**Running SITL**

 **1. Run SITL Setup**

    chmod +x /home/luke/Documents/Thesis/DroneAutoLander/SITL_Script.sh
    /home/luke/Documents/Thesis/DroneAutoLander/SITL_Script.sh

    /home/luke/Documents/Thesis/DroneAutoLander/SITL_Script_Batch_Runner.sh main
    SPEEDS="0" TRAJECTORIES="straight" /home/luke/Documents/Thesis/DroneAutoLander/SITL_Script_Batch_Runner.sh main

Then in the ardupilot terminal, run

    set streamrate 40

This now runs what used to be the following steps (a-d).

 **1a Run Gazebo Simulation**

    export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
    cd ardupilot_gazebo/worlds
    gz sim iris_runway_new.sdf -v -r

This will launch the gazebo simulation for the given world name (in this case `iris_runway_new.sdf`).
If models are edited, this will need to be refreshed.

**1b. Run ArduPilot SITL**

    cd ~/ardupilot && sim_vehicle.py -v ArduCopter \
    --console --map \
    -f JSON \
    --add-param-file=$HOME/ardupilot_gazebo/config/gazebo-iris-gimbal_1d.parm \
    --out=udp:127.0.0.1:14555

Starts the ArduPilot quad-copter SITL, loads the required params from gazebo-iris-gimbal.parm

**1c. Modify Ardupilot Settings**

The following settings should be changed on Ardupilot to improve MAVROS Publishing rates:

*In LINK:*

> Stream rate Link 1: 50.0

 (this will stop ArduPilot from overwriting the new parameters with default parameters)

**1d. Run MAVROS**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 run mavros mavros_node \
    --ros-args \
    -p use_sim_time:=true \
    -p fcu_url:=udp://:14555@ \
    --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_config.yaml \
    --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_pluginlists.yaml \
    -p send.tf:=true

Runs the MAVROS node which converts mavlink messages to ROS2 to enable communication between ArduPilot and ROS2 nodes.

> *NOTE:* This bottom MAVROS setup is not preferred because it does not generate the map → base_link tf2 transform. The top command runs off
> the `apm_config.yaml` (Ardupilot) parameter list, where importantly
> under global_position, `send_tf: true`
>
>     source /opt/ros/humble/setup.bash
>     cd ros2_ws
>     source install/setup.bash
>     ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14555@


**2a. Rebuild the launch file**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    colcon build --packages-select auto_lander

**2b. Run the launch file**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 launch auto_lander main_launch_sim.py

Starts all the required ROS2 nodes for the program to function (includes the GZ Bridge now). Add `record_bag:=true` to record a rosbag of the key topics.

**2c. Run Camera Calibration (only need to do once)**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 run auto_lander camera_calibrate

Only needed to calibrate camera once to determine camera matrix.

**3. Run Gazebo Rover Bridge (if you want to move rover)**

    gz topic -t "/cmd_rover_vel" -m gz.msgs.Twist -p "linear: {x: 0.5}, angular: {z: 0.5}"

This will move the rover with the given linear and angular velocity commands. These are in m/s and can be changed to suit whatever is needed.

**Extra**

    ros2 service call /mavros/cmd/command   mavros_msgs/srv/CommandLong   '{command: 205, param1: 0.0, param2: 0.0, param3: 0.0, param4: 0.0, param5: 0.0, param6: 0.0, param7: 1.0}'

Manual gimbal command to set to neutral mode
