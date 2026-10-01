# DroneAutoLander

## **Resources**
https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html

## Running IRL
**General Notes**

Before running in real-world, be sure to check the follow parameters are set up correctly:
1. Check the quad_to_cam frame transforms match that of the real world quadcopter setup, this includes both translation and rotation of the camera.
2. Check the AprilTag definitions, locations, and sizing match
3. Check the gains on the gimbal servo are ok/suitable
4. Adjust the measurement noise (R) based on quality of image, it will likely be worse than the simulation so it is likely that R will need to go up
5. Disable sim_time usage and any ground_truth diagnostics in main_launch.py
6. Disable gz_bridge and agv_controller from main_launch.py
7. Switch camera to webcam mode (need to check if this works) - get actual camera parameters!
8. Ensure that mavlink is outputting mavros messages at greater than 20Hz - if less, the quad will be very unstable!!
9. Run main_launch.py

Nearly all relevant params are located/exposed in the main_launch.py file. This will run all the relevant nodes needed automatically.
 
*Preparing for IRL testing:*
 
1. `use_sim_time` is `True` on **all four** nodes (apriltag, yolo, gimbal_controller, controller) and `--use-sim-time` is on the bag recorder. Set all to `False` (and remove `-p use_sim_time:=true` from the MAVROS command). If any node is missed, it will wait on a `/clock` that doesn't exist.
2. `ground_truth_available` → `False`.
3. `image_source` (AprilTag node) → `"webcam"`. The YOLO node might not have a `image_source`/webcam param in the launch file, so confirm that they do get the nodes do get the real camera feed (e.g. via a camera driver topic) before flying.
4. `show_debug_window`, `create_video`, `save_frames` → off (or at least `show_debug_window: False`) on a headless companion computer; they cost CPU and need a display. If recording video, set `output_dir` to a real writable path (currently `""`).
*Camera (apriltag_node, yolo_node, gimbal_controller)*
 
1. `camera_fov_horizontal` (2.7925268 rad ≈ 160°) is the simulated camera. Replace with the real lens value.
2. `camera_distortion_coeffs` are all zeros (no distortion in sim). Use the output of `camera_calibrate` at the **same resolution** you will run at, and recalibrate if resolution, focus or lens changes. A very wide lens may need a fisheye model.
3. `img_width`/`img_height` (960x540 AprilTag, 640x384 YOLO) must match the resolution that was calibrated and what the camera can deliver (same aspect ratio - no need to change). `webcam_fps` (30) should match `frame_capture_rate` (24).
*AprilTag (apriltag_node) - CAN IGNORE FOR NOW*
 
1. `apriltag_tag_ids`, `apriltag_spacing_m` (0.341), `apriltag_main_tag_size_m` (0.481), `apriltag_small_tag_size_m` (0.072), `apriltag_tag_to_pad_yaw_deg` (90) must match the **printed** pad. Measure the black square edge to edge (excluding the white border). A size error shows up directly as a range/height error.
2. `apriltag_main_offset_m` is `-0.0912 + 0.15`. The `+0.15` looks like a sim-specific tweak, so re-measure the real offset from main tag to pad centre.
3. `apriltag_quad_decimate` (2.0) may drop the 7.2 cm tag at altitude. Retune decimate/sigma/refine/sharpening on real footage, balancing detection range against CPU load.
*YOLO (yolo_node)*
 
1. `yolo_model_path` was trained on sim imagery and probably won't transfer well. Need to retrain/fine-tune on real pad (and vehicle) images, then re-check `yolo_landing_pad_class_id` (1), `yolo_car_class_id` (0) and `yolo_conf_threshold` (0.60) against the new model.
2. The model is an **OpenVINO** export (Intel CPU/iGPU). If the companion computer is a Jetson/Pi/other ARM board, re-export (TensorRT/NCNN/ONNX) and re-measure whether `yolo_processing_rate` (10 Hz) is achievable.
3. `yolo_ground_z` (1.5) is the assumed ground plane height in the local frame, i.e. how tall the car is above the ground. IMPORTANT -> Keep consistent with `landing_height_above_gnd` (1.5) parameter.
*Gimbal (gimbal_controller)*
 
1. Check the gimbal has position feedback, according to SIYI manual it seems like it does send it back? If it doesnt, then will need to change to use the commanded angle instead.
2. `gimbal_servo_min_angle_deg` / `gimbal_servo_max_angle_deg` (-90 / 25) must match the real mechanical travel **and** the servo/mount params on the real flight controller - think this matches so OK.
3. `gimbal_kp` (0.03), `gimbal_kd` (0.005), `gimbal_max_slew_deg_s` (60): the sim servo is ideal. A real servo has finite speed, backlash and deadband, so start low and watch for hunting/jitter.
4. `camera_offset_x/y/z` and `camera_mount_roll/pitch/yaw` (-π/2 for roll and yaw) must be measured on the real mount (see item 1).
*Controller – safety, geometry, landing logic*
 
1. `boundary_limit` (600 m) is not a real boundary for a test site. Set it to a small safe radius and also configure the ArduPilot geofence. `max_runtime` (200 s) should be checked against real battery endurance and what the vehicle does when it expires.
2. `gps_lat` / `gps_lon` are the SITL default location. IMPORTANT -> Set to the real loiter/waiting point. `gps_in_loc_buffer` (2.0 m) is the buffer room, so increase if its too tight".
3. `landing_error_threshold` (0.10 m) is the allowable error on landing from the estimated target centre. Also review `landing_height_above_gnd` (1.5), `landing_height_threshold` (0.4), `landing_chase_altitude` (6.0), `landing_recovery_height` (2.0), the descent rates (-1.0 / -0.5 m/s) and the locked/lost/visual-min timers. For first flights use higher altitudes, slower descents and larger margins, and confirm what actually happens at the end of the landing (mode switch / disarm / disengage).
*UKF*
 
1. R is **not** exposed in the launch file (hardcoded). Need to be retuned based on real-world images/video feed.
*PID / vehicle model*
 
1. `PID_mass` (1.98 kg), `PID_max_thrust` (40 N), `PID_gravity` and `PID_drag_coefficient` must reflect the real all-up weight (battery, gimbal, companion computer, camera) and the real thrust curve. With the current numbers hover is ≈ 1.98 × 9.81 / 40 ≈ 0.49 normalised thrust, so confirm the real vehicle hovers near the value this model predicts.
2. `PID_Kp_0`, `PID_Kd_0`, `PID_lam_0` and the z gains were tuned against an ideal sim attitude response. Start well below the SITL values and ramp up. Keep `PID_max_throttle_rate` and `PID_max_angle_rate` (1.0) conservative at first.
3. Terminal-phase shaping (`PID_terminal_gain`, `PID_drop_off`, `PID_drop_off_strength`, `PID_d_blend_*`, `PID_d_hold_radius`) is the part most sensitive to real-world effects (ground effect, wind, latency). Test it last and from altitude.
**Other things to be careful about**
 
1. **SITL param file:** `gazebo-iris-gimbal_1d.parm` is loaded into SITL via `--add-param-file`. None of those params exist on the real flight controller unless you set them (gimbal/servo/mount, guided options, etc.). Set as follows:

```text
    **Iris is X frame**
    FRAME_CLASS      1
    FRAME_TYPE       1

    **Match servo output for motors**
    MOT_PWM_MIN      1100
    MOT_PWM_MAX      1900

    **Gimbal/Mount**
    MNT1_TYPE        1
    MNT1_PITCH_MAX   25
    MNT1_PITCH_MIN   -90

    RC7_MAX          1900
    RC7_MIN          1100
    RC7_OPTION       213
   
    SERVO10_FUNCTION 7
    SERVO10_MIN      1100
    SERVO10_MAX      1900
```

2. **Stream rates:** `set streamrate 40` and the Link stream rate are SITL-only. On the real FCU set the `SRx_*` params for the telemetry port and use a high baud rate (e.g. 921600) for the MAVROS `fcu_url`. Verify with `ros2 topic hz` (see item 8) -> this is important because anything less than 20Hz results in a crash!
3. **Safety setup:** Would be good to have a manual override somewhere in the FCU configured.
4. **Compute budget:** run all nodes together and check actual rates (YOLO, AprilTag, UKF, controller at 20 Hz) and CPU/thermal load on the companion computer. Missed rates make the quad unstable (same as item 8) - check this.


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

Runs the MAVROS node which converts mavlink messages to ROS2  to enable communication between ArduPilot and ROS2 nodes. 

> *NOTE:* This bottom MAVROS setup is not preferred because it does not  generate the map → base_link tf2 transform. The top command runs off
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


Starts all the required ROS2 nodes for the program to function (includes the GZ Bridge now). This launches several nodes which can be examined in the source code.

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
