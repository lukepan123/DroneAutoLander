#ifndef CONTROLLER_STATE_HPP
#define CONTROLLER_STATE_HPP

#include "rclcpp/rclcpp.hpp"

#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include "mavros_msgs/srv/command_tol.hpp"
#include "mavros_msgs/srv/message_interval.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "std_msgs/msg/bool.hpp"

#include "tf2_ros/buffer.hpp"
#include "tf2_ros/transform_listener.hpp"

class ControllerState : public rclcpp::Node 
{
public:
    ControllerState();   // Constructor declaration only

private:
    // ---- CALLBACK FUNCTIONS ----
    void stateCallback(const mavros_msgs::msg::State::SharedPtr msg);
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void twistCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void targetFoundCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void targetLockedCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void requestMode();
    void onSetModeDone(rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future);
    void requestArm();
    void onArmDone(rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedFuture future);
    void requestTakeoff();
    void onTakeoffDone(rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture future);
    void initiateRTL(const std::string reason);
    void onRTLDone(rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future);
    void controlLoopTimerCallback();
    void safetyLoopTimerCallback();

    // ---- MEMBER VARIABLES ----
    // Global State Variables
    int controller_state_;
    const double max_runtime_;
    const double boundary_limit_;
    mavros_msgs::msg::State state_;
    nav_msgs::msg::Odometry odometry_;
    nav_msgs::msg::Odometry target_odometry_;

    // ---- State 0 (Pre-arm) Variables ----
    bool mode_requested_;
    bool mode_confirmed_;
    const std::string mode_;
    bool arm_requested_;
    bool arm_confirmed_;
    double armed_time_;

    // ---- State 1 (Take-off) Variables ----
    bool tko_requested_;
    bool tko_confirmed_;
    bool tko_wait_logged_;
    double tko_complete_time_;
    double tko_target_altitude_;

    // ---- State 2 (Searching for Target) Variables ----
    bool target_found_;
    bool target_locked_;

    // ---- State 3 (Maintaining Target Lock) Variables ---- 

    // ---- State 4 (Beginning Landing Descent) Variables ---- 

    // ---- State 5 (Landing Confirmed) Variables ---- 

    // ---- State 6 (Landing Abort) Variables ---- 

    // ---- State 7 (RTL) Variables ----
    bool rtl_initiated_;

    // ---- SUBSCRIPTIONS ----
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr fcu_state_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;

    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr target_found_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr target_locked_sub_;

    // ---- TF2 ----
    std::shared_ptr<tf2_ros::Buffer> tf_map_target_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_map_target_listener_;

    // ---- CLIENTS ----
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr takeoff_client_;
    rclcpp::Client<mavros_msgs::srv::MessageInterval>::SharedPtr message_interval_client_;

    // ---- CONTROL TIMERS ----
    rclcpp::TimerBase::SharedPtr control_loop_timer_;
    rclcpp::TimerBase::SharedPtr safety_loop_timer_;
};

#endif // CONTROLLER_STATE_HPP