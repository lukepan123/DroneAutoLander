# DroneAutoLander

## **Resources**
https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html

## Running
**General Notes**

Each application will need to be run in separate terminal windows.

If after running `source install/setup.bash` no ROS2 commands are found, you will also need to run `source /opt/ros/humble/setup.bash`

To rebuild package:

    colcon build --packages-select auto_lander

**Running SITL**

 **1. Run Gazebo Simulation**

    export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
    cd ardupilot_gazebo/worlds
    gz sim iris_runway_new.sdf -v -r

This will launch the gazebo simulation for the given world name (in this case `iris_runway_new.sdf`). 
If models are edited, this will need to be refreshed.

**2a. Run Gazebo Camera + Gimbal Bridge**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 run ros_gz_bridge parameter_bridge \
    /world/iris_runway_new/model/iris_with_gimbal/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
    /model/LandingVehicle/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
    /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock --ros-args -r /world/iris_runway_new/model/iris_with_gimbal/link/camera_link/sensor/camera/image:=/camera/image_raw \
    -r "/model/LandingVehicle/odometry:=/landing_pad/odom"

This will run the gazebo camera and gimbal bridge, linking the gazebo camera images to the camera ROS2 node and gimbal topics.

**3a. Run ArduPilot SITL**

    cd ~/ardupilot && sim_vehicle.py -v ArduCopter --console --map -w --out=udp:127.0.0.1:14555 -f gazebo-iris --model JSON

Starts the ArduPilot quad-copter SITL. 

**3b. Modify Ardupilot Settings**

The following settings should be changed on Ardupilot to improve MAVROS Publishing rates:

*In LINK:*

> Stream rate Link 1: 50.0

 (this will stop ArduPilot from overwriting the new parameters with default parameters)

**4. Run MAVROS**

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


**5a. Run the launch file**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 launch auto_lander main_launch.py

Starts all the required ROS2 nodes for the program to function. This launches several nodes which can be examined in the source code.

**5b. Run Camera Calibration (only need to do once)**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 run auto_lander camera_calibrate

Only needed to calibrate camera once to determine camera matrix.

**6. Run Gazebo Rover Bridge (if you want to move rover)**

    gz topic -t "/cmd_rover_vel" -m gz.msgs.Twist -p "linear: {x: 0.5}, angular: {z: 0.5}"

This will move the rover with the given linear and angular velocity commands. These are in m/s and can be changed to suit whatever is needed.
