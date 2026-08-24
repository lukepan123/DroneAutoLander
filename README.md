# DroneAutoLander

## **Resources**
https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html

## Running IRL
**General Notes**

Before running in real-world, be sure to check the follow parameters are set up correctly:
1. Check the _compose_base_to_landing_pad frame transforms match that of the real world quadcopter setup, this includes both translation and rotation of the camera.
2. Check the AprilTag definitions, locations, and sizing match
3. Check the gains on the gimbal servo are ok/suitable
4. Adjust the measurement noise (R) based on quality of image, it will likely be worse than the simulation so it is likely that R will need to go up
5. Disable sim_time usage and any ground_truth diagnostics in main_launch.py
6. Disable gz_bridge and agv_controller from main_launch.py
7. Switch camera to webcam mode (need to check if this works) - get actual camera parameters!
8. Ensure that mavlink is outputting mavros messages at greater than 20Hz - if less, the quad will be very unstable!!

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
    ros2 launch auto_lander main_launch.py

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
