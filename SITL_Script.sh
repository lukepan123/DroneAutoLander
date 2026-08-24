#!/bin/bash

# Tab 1: Gazebo
gnome-terminal --tab -- bash -c '
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
cd ~/ardupilot_gazebo/worlds
gz sim iris_runway_new.sdf -v -r
exec bash
'

sleep 5

# Tab 2: ArduPilot SITL
gnome-terminal --tab -- bash -c '
cd ~/ardupilot
sim_vehicle.py -v ArduCopter \
    --console --map \
    -f JSON \
    --add-param-file=$HOME/ardupilot_gazebo/config/gazebo-iris-gimbal_1d.parm \
    --out=udp:127.0.0.1:14555 \
    --mavproxy-args="--cmd=\"set streamrate 40\""
exec bash
'

sleep 10

# Tab 3: MAVROS
gnome-terminal --tab -- bash -c '
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
source install/setup.bash

ros2 run mavros mavros_node \
    --ros-args \
    -p use_sim_time:=true \
    -p fcu_url:=udp://:14555@ \
    --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_config.yaml \
    --params-file $(ros2 pkg prefix mavros)/share/mavros/launch/apm_pluginlists.yaml \
    -p send.tf:=true

exec bash
'