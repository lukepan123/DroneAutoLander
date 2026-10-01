import sys
import signal
import numpy as np
import tf2_ros
import rclpy
import tf_transformations

from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import TransformStamped
from mavros_msgs.msg import GimbalManagerSetPitchyaw
from mavros_msgs.msg import GimbalDeviceAttitudeStatus

from .vision_common import CameraMount
from .vision_common import LatestValueBuffer
from .vision_common import camera_pose_in_level_frame
from .vision_common import image_target_to_angle_error
from .vision_common import stamp_to_sec

""" Gimbal Controller Node.

    Owns the only gimbal actuator independently of the AprilTag and YOLO perception
    nodes. Perception nodes publish target points in their image and this node decides
    where the camera should point using the same image geometry for either source.

    AprilTag is the priority target whenever its target is still fresh. YOLO is used
    as a fallback when AprilTag is unavailable or its target has gone stale. If both
    are stale, the gimbal simply holds its last commanded angle.

    This node also broadcasts the camera pose as a TF transform:
        local -> gimbal_camera_optical_frame

    The translation is the fixed camera offset relative to the vehicle and the
    rotation uses the vehicle attitude plus the measured gimbal pitch. AprilTag and
    YOLO use this TF to construct their own landing-pad measurement transforms.
"""


class GimbalControllerNode(Node):
    """ Defines the independent gimbal controller node. """

    def __init__(self) -> None:
        """ Initialise the independent gimbal controller node. """

        super().__init__("gimbal_controller")

        # ---- PROCESSING PARAMETERS ----
        self.declare_parameter("gimbal_control_rate", 20.0)
        self._gimbal_control_rate = (
            self.get_parameter("gimbal_control_rate")
            .get_parameter_value()
            .double_value
        )

        # ---- TARGET PARAMETERS ----
        # How old a buffered target is allowed to be before it is treated as stale.
        self.declare_parameter("gimbal_apriltag_target_max_age_s", 0.15)
        self._gimbal_apriltag_target_max_age_s = (
            self.get_parameter("gimbal_apriltag_target_max_age_s")
            .get_parameter_value()
            .double_value
        )

        self.declare_parameter("gimbal_yolo_target_max_age_s", 0.5)
        self._gimbal_yolo_target_max_age_s = (
            self.get_parameter("gimbal_yolo_target_max_age_s")
            .get_parameter_value()
            .double_value
        )

        # ---- GIMBAL CONTROLLER PARAMETERS ----
        self.declare_parameter("gimbal_kp", 0.03)
        self.declare_parameter("gimbal_kd", 0.005)
        self.declare_parameter("gimbal_max_slew_deg_s", 60.0)
        self.declare_parameter("gimbal_initial_angle_deg", -90.0)
        self.declare_parameter("gimbal_actual_pitch_initial_deg", -90.0)
        self.declare_parameter("gimbal_servo_min_angle_deg", -90.0)
        self.declare_parameter("gimbal_servo_max_angle_deg", 25.0)
        self.declare_parameter("gimbal_yaw_command_deg", 0.0)

        self._gimbal_Kp = (
            self.get_parameter("gimbal_kp").get_parameter_value().double_value
        )
        self._gimbal_Kd = (
            self.get_parameter("gimbal_kd").get_parameter_value().double_value
        )
        self._gimbal_max_slew_deg_s = (
            self.get_parameter("gimbal_max_slew_deg_s")
            .get_parameter_value()
            .double_value
        )
        self._servo_angle = (
            self.get_parameter("gimbal_initial_angle_deg")
            .get_parameter_value()
            .double_value
        )
        self._gimbal_actual_pitch = (
            self.get_parameter("gimbal_actual_pitch_initial_deg")
            .get_parameter_value()
            .double_value
        )
        self._servo_min_angle = (
            self.get_parameter("gimbal_servo_min_angle_deg")
            .get_parameter_value()
            .double_value
        )
        self._servo_max_angle = (
            self.get_parameter("gimbal_servo_max_angle_deg")
            .get_parameter_value()
            .double_value
        )
        self._gimbal_yaw_command_deg = (
            self.get_parameter("gimbal_yaw_command_deg")
            .get_parameter_value()
            .double_value
        )

        self._gimbal_prev_error = 0.0
        self._gimbal_last_cmd_time = None
        self._gimbal_last_source = None

        # ---- CAMERA PARAMETERS ----
        self.declare_parameter("camera_offset_x", 0.02)
        self.declare_parameter("camera_offset_y", -0.01)
        self.declare_parameter("camera_offset_z", -0.124923)
        self.declare_parameter("camera_mount_roll", -1.5707963)
        self.declare_parameter("camera_mount_pitch", 0.0)
        self.declare_parameter("camera_mount_yaw", -1.5707963)

        self._camera_mount = CameraMount(
            offset_x=(
                self.get_parameter("camera_offset_x")
                .get_parameter_value()
                .double_value
            ),
            offset_y=(
                self.get_parameter("camera_offset_y")
                .get_parameter_value()
                .double_value
            ),
            offset_z=(
                self.get_parameter("camera_offset_z")
                .get_parameter_value()
                .double_value
            ),
            roll_offset=(
                self.get_parameter("camera_mount_roll")
                .get_parameter_value()
                .double_value
            ),
            pitch_offset=(
                self.get_parameter("camera_mount_pitch")
                .get_parameter_value()
                .double_value
            ),
            yaw_offset=(
                self.get_parameter("camera_mount_yaw")
                .get_parameter_value()
                .double_value
            ),
        )

        self.get_logger().info(
            f"Gimbal controller rate: {self._gimbal_control_rate} Hz"
        )
        self.get_logger().info(
            f"Gimbal target ages: apriltag={self._gimbal_apriltag_target_max_age_s}s, "
            f"yolo={self._gimbal_yolo_target_max_age_s}s"
        )

        # ---- SUBSCRIPTIONS ----
        _odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._quad_odometry_sub = self.create_subscription(
            Odometry,
            "/mavros/global_position/local",
            self._quad_odometry_callback,
            _odom_qos,
        )
        self._gimbal_attitude_sub = self.create_subscription(
            GimbalDeviceAttitudeStatus,
            "/mavros/gimbal_control/device/attitude_status",
            self._gimbal_attitude_callback,
            10,
        )
        self._apriltag_target_sub = self.create_subscription(
            Vector3Stamped,
            "/landing_pad/apriltag_target",
            self._apriltag_target_callback,
            10,
        )
        self._yolo_target_sub = self.create_subscription(
            Vector3Stamped,
            "/landing_pad/yolo_target",
            self._yolo_target_callback,
            10,
        )

        self._apriltag_target_buffer = LatestValueBuffer()
        self._yolo_target_buffer = LatestValueBuffer()
        self._quad_rotation = None

        # ---- PUBLISHERS ----
        self._gimbal_angle_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/gimbal_angle", 10
        )
        self._gimbal_manager_publisher = self.create_publisher(
            GimbalManagerSetPitchyaw,
            "/mavros/gimbal_control/manager/set_pitchyaw",
            10,
        )

        # ---- TF2 ----
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- PROCESSING TIMER ----
        self._gimbal_timer = self.create_timer(
            1.0 / self._gimbal_control_rate, self._gimbal_timer_callback
        )

    # ---- RECEPTION CALLBACKS ----
    def _quad_odometry_callback(self, msg: Odometry) -> None:
        """ Obtain the latest FCU attitude for the camera TF.

        :param msg: Incoming Odometry message
        """
        self._quad_rotation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]

    def _gimbal_attitude_callback(self, msg: GimbalDeviceAttitudeStatus) -> None:
        """Store the actual measured gimbal pitch from the attitude quaternion."""

        q = [msg.q.x, msg.q.y, msg.q.z, msg.q.w,]
        _, pitch, _ = tf_transformations.euler_from_quaternion(q)

        self._gimbal_actual_pitch = float(np.degrees(pitch))

        # self.get_logger().info(
        #     f"Gimbal actual pitch: {self._gimbal_actual_pitch:.2f} deg",
        #     throttle_duration_sec=1.0,
        # )

    def _apriltag_target_callback(self, msg: Vector3Stamped) -> None:
        """ Buffer an AprilTag image target for the independent gimbal controller.

        :param msg: Vector3Stamped target. vector.x/y are normalised image coordinates
                    and vector.z is the image vertical FOV in radians.
        """
        self._apriltag_target_buffer.push(msg, stamp_to_sec(msg.header.stamp))

    def _yolo_target_callback(self, msg: Vector3Stamped) -> None:
        """ Buffer a YOLO image target for fallback gimbal control.

        :param msg: Vector3Stamped target. vector.x/y are normalised image coordinates
                    and vector.z is the image vertical FOV in radians.
        """
        self._yolo_target_buffer.push(msg, stamp_to_sec(msg.header.stamp))

    def _get_target(self):
        """ Return the freshest target, prioritising AprilTag over YOLO. """
        t_now = self.get_clock().now().nanoseconds / 1e9

        apriltag_target = self._apriltag_target_buffer.get_if_fresh(
            t_now, self._gimbal_apriltag_target_max_age_s
        )
        if apriltag_target is not None:
            return apriltag_target, "apriltag"

        yolo_target = self._yolo_target_buffer.get_if_fresh(
            t_now, self._gimbal_yolo_target_max_age_s
        )
        if yolo_target is not None:
            return yolo_target, "yolo"

        return None, None

    def _gimbal_timer_callback(self) -> None:
        """ Run the independent gimbal controller and broadcast the camera TF. """

        target, source = self._get_target()
        if target is not None:
            angle_error = image_target_to_angle_error(
                float(target.vector.y),
                float(target.vector.z),
            )
            self._drive_gimbal(angle_error, source) # type: ignore

        self._gimbal_manager_control(self._servo_angle)
        self._publish_camera_tf()
        self._publish_gimbal_angle()

    def _drive_gimbal(self, angle_error: float, source: str) -> None:
        """ Feed an angle error into the gimbal PD controller (_gimbal_controller),
            resetting the derivative term's baseline whenever the error's source
            switches between "apriltag" and "yolo". The two pipelines are
            independent sensors on different cadences/resolutions, so their error
            signals aren't continuous with each other - differentiating straight
            across a switch would produce a spurious derivative kick in the servo
            command.

        :param angle_error: Signed vertical angle error (degrees) - see
                             image_target_to_angle_error in vision_common.py
        :param source:      "apriltag" or "yolo", identifying which pipeline this
                             error came from
        """
        if self._gimbal_last_source is not None and self._gimbal_last_source != source:
            self._gimbal_prev_error = angle_error
        self._gimbal_last_source = source
        self._gimbal_controller(angle_error)

    def _gimbal_controller(self, angle_error: float) -> None:
        """ PD control law that steps the commanded servo angle towards centring the
            current target in the image row.

        :param angle_error: Signed vertical angle error (degrees) of the target from
                             the image row centre; positive = target below centre
        """

        now = self.get_clock().now()
        if self._gimbal_last_cmd_time is None:
            dt = 1.0 / self._gimbal_control_rate
        else:
            dt = (now - self._gimbal_last_cmd_time).nanoseconds / 1e9
            dt = max(dt, 1e-3)
        self._gimbal_last_cmd_time = now

        derivative = (angle_error - self._gimbal_prev_error) / dt
        self._gimbal_prev_error = angle_error

        correction = (
            self._gimbal_Kp * angle_error
            + self._gimbal_Kd * derivative
        )
        max_step = self._gimbal_max_slew_deg_s * dt
        correction = np.clip(correction, -max_step, max_step)

        self._servo_angle = np.clip(
            self._servo_angle - correction, self._servo_min_angle, self._servo_max_angle
        )

    def _gimbal_manager_control(self, pitch_angle_deg: float):
        msg = GimbalManagerSetPitchyaw()
        msg.pitch = float(np.deg2rad(pitch_angle_deg))  # check units - some mavros versions want rad, some deg; verify against `ros2 interface show`
        msg.yaw = float(np.deg2rad(self._gimbal_yaw_command_deg))  # see NaN note below
        msg.pitch_rate = float("nan")
        msg.yaw_rate = float("nan")
        self._gimbal_manager_publisher.publish(msg)

    def _publish_gimbal_angle(self) -> None:
        """ Publish the actual gimbal angle every controller tick. """
        gimbal_msg = Vector3Stamped()
        gimbal_msg.header.stamp = self.get_clock().now().to_msg()
        gimbal_msg.header.frame_id = "gimbal_angle"
        gimbal_msg.vector.x = float(self._gimbal_actual_pitch)
        self._gimbal_angle_publisher.publish(gimbal_msg)

    def _publish_camera_tf(self) -> None:
        """ Broadcast local -> gimbal_camera_optical_frame using actual gimbal pitch. """
        if self._quad_rotation is None:
            return

        T_local_cam = camera_pose_in_level_frame(
            self._quad_rotation,
            self._gimbal_actual_pitch,
            self._camera_mount,
        )
        t_out = T_local_cam[:3, 3]
        q_out = tf_transformations.quaternion_from_matrix(T_local_cam)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "local"
        tf_msg.child_frame_id = "gimbal_camera_optical_frame"
        tf_msg.transform.translation.x = float(t_out[0])
        tf_msg.transform.translation.y = float(t_out[1])
        tf_msg.transform.translation.z = float(t_out[2])
        tf_msg.transform.rotation.x = float(q_out[0])
        tf_msg.transform.rotation.y = float(q_out[1])
        tf_msg.transform.rotation.z = float(q_out[2])
        tf_msg.transform.rotation.w = float(q_out[3])

        self._tf_broadcaster.sendTransform(tf_msg)


# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = GimbalControllerNode()

    shutdown_in_progress = False

    def signal_handler(signum, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            node.get_logger().warn("Second interrupt received! Force terminating...")
            sys.exit(1)
        else:
            shutdown_in_progress = True
            node.get_logger().info("Interrupt received, shutting down...")
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
