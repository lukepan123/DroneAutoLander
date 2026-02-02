#include "controller_state_orchestrator.hpp"

using std::placeholders::_1;

ControllerState::ControllerState()
: Node("controller_state"),
    // Global State Variables
    controller_state_(0),
    max_runtime_(90.0),
    boundary_limit_(30.0),
    state_(),
    odometry_(),
    target_odometry_(),

    // ---- State 0 (Pre-arm) Variables ----
    mode_requested_(false),
    mode_confirmed_(false),
    mode_("GUIDED"),
    arm_requested_(false),
    arm_confirmed_(false),
    armed_time_(0.0),

    // ---- State 1 (Take-off) Variables ----
    tko_requested_(false),
    tko_confirmed_(false),
    tko_wait_logged_(false),
    tko_complete_time_(0.0),
    tko_target_altitude_(15.0),

    // ---- State 2 (Searching for Target) Variables ----
    target_found_(false),
    target_locked_(false),

    // ---- State 3 (Maintaining Target Lock) Variables ---- 

    // ---- State 4 (Beginning Landing Descent) Variables ---- 

    // ---- State 5 (Landing Confirmed) Variables ---- 

    // ---- State 6 (Landing Abort) Variables ---- 

    // ---- State 7 (RTL) Variables ----
    rtl_initiated_(false)
{
    // ---- SUBSCRIPTIONS ----
    fcu_state_sub_ = create_subscription<mavros_msgs::msg::State>(
        "/mavros/state",
        10,
        std::bind(&ControllerState::stateCallback, this, _1)
    );

    odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/mavros/global_position/local",
        10,
        std::bind(&ControllerState::odometryCallback, this, _1)
    );

    twist_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/mavros/local_position/velocity_local",
        10,
        std::bind(&ControllerState::twistCallback, this, _1)
    );

    // ---- TF2 ----
    tf_map_target_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_map_target_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_map_target_buffer_);

    // ---- CLIENTS ----
    set_mode_client_ = create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");
    arming_client_ = create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
    takeoff_client_ = create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");
    message_interval_client_ = create_client<mavros_msgs::srv::MessageInterval>("/mavros/set_message_interval");

    // ---- CONTROL TIMERS
    control_loop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), // 20 ms = 50 Hz
            std::bind(&ControllerState::controlLoopTimerCallback, this)
        );
    safety_loop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000), // 1000 ms = 1 Hz
            std::bind(&ControllerState::safetyLoopTimerCallback, this)
        );
}

// ---- CALLBACK IMPLEMENTATIONS ----
void ControllerState::stateCallback(const mavros_msgs::msg::State::SharedPtr msg) 
{ 
    state_ = *msg;
    
    // Monitor and update FCU Mode
    if (state_.mode == mode_ && !mode_confirmed_) {
        mode_confirmed_ = true;
        RCLCPP_INFO(this->get_logger(), "%s confirmed by FCU.", mode_.c_str());
    }

    // Monitor and update armed status
    if (state_.armed == true && !arm_confirmed_) {
        arm_confirmed_ = true;
        armed_time_  = this->get_clock()->now().seconds();
    }
}

void ControllerState::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{ 
    
}

void ControllerState::twistCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{ 
    
}

void ControllerState::targetFoundCallback(const std_msgs::msg::Bool::SharedPtr msg)
{ 
    
}

void ControllerState::targetLockedCallback(const std_msgs::msg::Bool::SharedPtr msg)
{ 
    
}

void ControllerState::requestMode()
{
    // Check if service is available - if so send an FCU mode request
    if (!set_mode_client_->wait_for_service(std::chrono::milliseconds(500))) {
        RCLCPP_WARN(this->get_logger(), "SetMode service not ready yet.");
        return;
    }
    mode_requested_ = true;
    auto req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
    req->custom_mode = mode_;
    auto fut = set_mode_client_->async_send_request(
        req,
        std::bind(&ControllerState::onSetModeDone, this, std::placeholders::_1)
    );
    RCLCPP_INFO(this->get_logger(), "Requesting %s...", mode_.c_str());
}

void ControllerState::onSetModeDone(
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future)
{
    // Validate the FCU mode set was successful and write result if so
    try {
        auto res = future.get();
        if (res->mode_sent) {
            RCLCPP_INFO(this->get_logger(), "%s command accepted (awaiting FCU report).", mode_.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "%s command rejected by FCU.", mode_.c_str());
            mode_requested_ = false;
        }
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "SetMode exception: %s", e.what());
        mode_requested_ = false;
    }
}

void ControllerState::requestArm()
{
    // Check if service is available - if so send an FCU arm request
    if (!arming_client_->wait_for_service(std::chrono::milliseconds(500))) {
        RCLCPP_WARN(this->get_logger(), "Arming service not ready yet.");
        return;
    }
    arm_requested_ = true;
    auto req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
    req->value = true;
    auto future_result = arming_client_->async_send_request(
        req,
        std::bind(&ControllerState::onArmDone, this, std::placeholders::_1)
    );
    RCLCPP_INFO(this->get_logger(), "Requesting ARM...");
}

void ControllerState::onArmDone(
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedFuture future)
{
    // Validate the FCU arm request was successful and write result if so
    try {
        auto res = future.get();
        if (res->success) {
            RCLCPP_INFO(this->get_logger(), "Arm accepted (awaiting FCU armed=true).");
        } else {
            RCLCPP_ERROR(this->get_logger(),
                         "Arm rejected by FCU (result=%d)", res->success);
            arm_requested_ = false; // Reset request on fail
        }
    } 
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Arming exception: %s", e.what());
        arm_requested_ = false; // Reset request on fail
    }
}

void ControllerState::requestTakeoff()
{
    // Check if service is available - if so send FCU TKO Request to takeoff altitude
    if (!takeoff_client_->wait_for_service(std::chrono::milliseconds(500))) {
        RCLCPP_WARN(this->get_logger(), "Takeoff service not ready yet.");
        return;
    }
    tko_requested_ = true;
    auto req = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
    req->altitude = tko_target_altitude_;
    auto fut = takeoff_client_->async_send_request(
        req,
        std::bind(&ControllerState::onTakeoffDone, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Requesting takeoff to %.1f m...", tko_target_altitude_);
}

void ControllerState::onTakeoffDone(
    rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture future)
{
    // Validate the FCU takeoff request was successful and write result if so
    try {
        auto res = future.get();
        if (res->success) {
            RCLCPP_INFO(this->get_logger(), "Takeoff command accepted (monitoring altitude).");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Takeoff rejected by FCU.");
            tko_requested_ = false; // Reset request on fail
        }
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Takeoff exception: %s", e.what());
        tko_requested_ = false; // Reset request on fail
    }
}

void ControllerState::initiateRTL(std::string reason) 
{
    // Initiate RTL and exit control loop
    if (rtl_initiated_) { return; }

    rtl_initiated_ = true;

    RCLCPP_WARN(this->get_logger(), "SAFETY: Initiating RTL due to %s", reason.c_str());

    if (!set_mode_client_->wait_for_service(std::chrono::milliseconds(1000))) {
        RCLCPP_ERROR(this->get_logger(), "SetMode service not available for RTL!");
        return;
    }

    auto req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
    req->custom_mode = "RTL";
    auto future = set_mode_client_->async_send_request(
        req,
        std::bind(&ControllerState::onRTLDone, this, std::placeholders::_1)
    );
}

void ControllerState::onRTLDone(
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future)
{
    // Validate the FCU RTL request and confirm
    try {
        auto res = future.get();

        if (res->mode_sent) {
            RCLCPP_INFO(this->get_logger(), "RTL command accepted");
        } else {
            RCLCPP_ERROR(this->get_logger(), "RTL command rejected by FCU");
        }

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "RTL exception: %s", e.what());
    }
}

void ControllerState::controlLoopTimerCallback() 
{
    // ---- (State 0 (Pre-arm) - Below conditions need to be met ALWAYS so we check regardless of state
    if (!state_.connected) { 
        controller_state_ = 0;
        return; 
    }

    // Try to set mode
    if (!mode_confirmed_) {
        if (!mode_requested_) {
            requestMode();
        }
        controller_state_ = 0;
        return;
    }

    // Try to arm
    if (!arm_confirmed_) {
        if (!arm_requested_) {
            requestArm();
        }
        controller_state_ = 0;
        return;
    }

    // ---- State 1 (Takeoff)
    if (arm_confirmed_ && armed_time_ > 0.0 && !tko_requested_ ) {
        controller_state_ = 1;
        double elapsed = this->get_clock()->now().seconds() - armed_time_;
        if (elapsed < 5.0) {
            if (!tko_wait_logged_) {
                RCLCPP_INFO(this->get_logger(), "Armed. Waiting 5s before takeoff...");
                tko_wait_logged_ = true;
            }
            return;
        } else {
            requestTakeoff();
            tko_wait_logged_ = false;
            return;
        }
        
    }

    // ---- State 2 (Searching for Target)
}

void ControllerState::safetyLoopTimerCallback() 
{
    // Check safety conditions and initiate RTL if necessary
    if (rtl_initiated_ || tko_confirmed_) {
        return;
    }

    double current_time = this->get_clock()->now().nanoseconds() / 1e9;

    // Check run-time timer after takeoff
    if (tko_complete_time_ > 0.0) {
        double elapsed_since_takeoff = current_time - tko_complete_time_;
        if (elapsed_since_takeoff >= max_runtime_) {
            RCLCPP_INFO(this->get_logger(), "%f seconds elapsed since takeoff - Initiating RTL", max_runtime_);
            initiateRTL("Timer expired");
            controller_state_ = 7;
            return;
        }
    }

    // Check boundary conditions (30x30m square)
    float x = odometry_.pose.pose.position.x;
    float y = odometry_.pose.pose.position.y;

    if (std::abs(x) > boundary_limit_ || std::abs(y) > boundary_limit_) {
        RCLCPP_WARN(this->get_logger(), "Boundary violation: position (%f, %f) - Initiating RTL", x, y);
        initiateRTL("Boundary violation");
        controller_state_ = 7;
        return;
    }

}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControllerState>());
    rclcpp::shutdown();
    return 0;
}