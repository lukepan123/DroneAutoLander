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

**2a. Run Gazebo Camera Bridge**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 run ros_gz_bridge parameter_bridge /world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image --ros-args -r /world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image:=/camera/image_raw

    ros2 run ros_gz_image image_bridge /world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image /camera/image_raw

This will run the gazebo camera bridge, linking the gazebo camera images to the camera ROS2 node.

**2b. Run Gazebo Gimbal Bridge**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 run ros_gz_bridge parameter_bridge \
    /gimbal/cmd_roll@std_msgs/msg/Float64@gz.msgs.Double \
    /gimbal/cmd_pitch@std_msgs/msg/Float64@gz.msgs.Double \
    /gimbal/cmd_yaw_vel@std_msgs/msg/Float64@gz.msgs.Double \
    /gimbal/yaw@sensor_msgs/msg/JointState@gz.msgs.Model

This will run the gazebo gimbal bridge, linking the ROS2 topics to the gazebo gimbal topics.

**3a. Run ArduPilot SITL**

    cd ~/ardupilot && sim_vehicle.py -v ArduCopter --console --map -w --out=udp:127.0.0.1:14555 -f gazebo-iris --model JSON

Starts the ArduPilot quad-copter SITL. 

**3b. Modify Ardupilot Settings**

The following settings should be changed on Ardupilot to improve MAVROS Publishing rates:

*In LINK:*

> Stream rate Link 1: 30.0
> 
> Stream rate Link 2: 30.0
> 
> Baud Rate of New Links: 115200

*In UNIT:*

> Try FTP for parameter download: false

 (this will stop ArduPilot from overwriting the new parameters with default parameters)

**4. Run MAVROS**

    source /opt/ros/humble/setup.bash
    cd ros2_ws
    source install/setup.bash
    ros2 run mavros mavros_node \
    --ros-args \
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
