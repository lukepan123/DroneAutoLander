# DroneAutoLander

Resources
https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html
Running
Each application will need to be run in separate terminal windows. 
If after running source install/setup.bash no ROS2 commands are found, you will also need to run source /opt/ros/humble/setup.bash
To rebuild package:
colcon build --packages-select auto_lander
1. Run Gazebo Simulation
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
cd ardupilot_gazebo/worlds
gz sim iris_runway_new.sdf -v -r
This will launch the gazebo simulation for the given world name (in this case “iris_runway.sdf).
2. Run Gazebo Camera Bridge
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
ros2 run ros_gz_bridge parameter_bridge /world/iris_runway_new/model/iris_with_gimbal/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image --ros-args -r /world/iris_runway_new/model/iris_with_gimbal/link/camera_link/sensor/camera/image:=/camera/image_raw
This will run the gazebo camera bridge, linking the gazebo camera images to the camera ROS2 node.
3a. Run ArduPilot
cd ~/ardupilot && sim_vehicle.py -v ArduCopter --console --map -w --out=udp:127.0.0.1:14555 -f gazebo-iris --model JSON
Starts the ArduPilot quad-copter command software.
3b. Modify Ardupilot Settings
The following settings should be changed on Ardupilot to improve MAVROS Publishing rates:
In LINK:
Stream rate Link 1: 30.0
Stream rate Link 2: 30.0
Baud Rate of New Links: 115200
In UNIT:
Try FTP for paramter download: false (this will stop ardupilot from overwriting the new params with default params)
4. Run MAVROS
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14555@
This bottom is preferred because it will automatically generate the map → base_link tf transform. It runs off the APM (Ardupilot) paramter list.
ros2 run mavros mavros_node \
  --ros-args \
  -p fcu_url:=udp://:14555@ \
  --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_config.yaml \
  --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_pluginlists.yaml \
  -p send.tf:=true
Runs the MAVROS mavlink to ROS2 conversion to enable communication between ardupilot and ROS2 nodes
5a. Run Camera Calibration (only need to do once)
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
ros2 run auto_lander camera_calibrate
5b. Run ArUco Perception
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
ros2 run auto_lander tagposedetector
Start the camera vision detection ROS2 node.
5c. Run Controller
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
ros2 run circumnavigation_controller controller
5d. Launch Controller

source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
ros2 launch auto_lander main_launch.py
Start the controller ROS2 node.
6. Run Gazebo Rover Bridge (if you want to run rover)
gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.5}, angular: {z: 0.05}"
This will move the rover with the given linear and angular velocity commands.

