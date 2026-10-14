import sys
import signal
import numpy as np
import rclpy
import tf_transformations

from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from mavros_msgs.msg import GimbalManagerSetPitchyaw
from mavros_msgs.msg import GimbalDeviceAttitudeStatus

from .vision_common import LatestValueBuffer
from .vision_common import image_target_to_angle_error
from .vision_common import stamp_to_sec

""" Gimbal Controller Node.

    Owns the only gimbal actuator independently of the AprilTag and YOLO perception
    nodes. Perception nodes publish target points in their image and this node decides
    where the camera should point using the same image geometry for either source.

    AprilTag is the priority target whenever its target is still fresh. YOLO is used
    as a fallback when AprilTag is unavailable or its target has gone stale. If both
    are stale, the gimbal simply holds its last commanded angle.

    The measured gimbal pitch is republished on /landing_pad/gimbal_angle, stamped at
    measurement time. The perception nodes interpolate it to each image stamp and build
    the camera pose themselves (vision_common.camera_pose_in_level_frame).
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
        self._gimbal_status_received = False

        self.get_logger().info(
            f"Gimbal controller rate: {self._gimbal_control_rate} Hz"
        )
        self.get_logger().info(
            f"Gimbal target ages: apriltag={self._gimbal_apriltag_target_max_age_s}s, "
            f"yolo={self._gimbal_yolo_target_max_age_s}s"
        )

        # ---- SUBSCRIPTIONS ----
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

        # ---- PUBLISHERS ----
        self._gimbal_angle_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/gimbal_angle", 10
        )
        self._gimbal_manager_publisher = self.create_publisher(
            GimbalManagerSetPitchyaw,
            "/mavros/gimbal_control/manager/set_pitchyaw",
            10,
        )

        # ---- PROCESSING TIMER ----
        self._gimbal_timer = self.create_timer(
            1.0 / self._gimbal_control_rate, self._gimbal_timer_callback
        )

    # ---- RECEPTION CALLBACKS ----
    def _gimbal_attitude_callback(self, msg: GimbalDeviceAttitudeStatus) -> None:
        """ Store the measured gimbal pitch and publish it stamped at measurement time,
            so the perception nodes interpolate the angle the camera actually had when
            each image was taken.
        """
        q = [msg.q.x, msg.q.y, msg.q.z, msg.q.w]
        _, pitch, _ = tf_transformations.euler_from_quaternion(q)
        self._gimbal_actual_pitch = float(np.degrees(pitch))
        self._gimbal_status_received = True

        # Fall back to receive time if MAVROS left the header unstamped
        stamp = msg.header.stamp
        if stamp_to_sec(stamp) <= 0.0:
            stamp = self.get_clock().now().to_msg()
        self._publish_gimbal_angle(stamp)

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
        """ Run the independent gimbal controller. """

        target, source = self._get_target()
        if target is not None:
            angle_error = image_target_to_angle_error(
                float(target.vector.y),
                float(target.vector.z),
            )
            self._drive_gimbal(angle_error, source) # type: ignore

        self._gimbal_manager_control(self._servo_angle)

        # Startup only: publish the initial angle until the first real status arrives,
        # so the perception nodes don't sit waiting for a gimbal angle
        if not self._gimbal_status_received:
            self._publish_gimbal_angle(self.get_clock().now().to_msg())

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

    def _publish_gimbal_angle(self, stamp) -> None:
        """ Publish the measured gimbal angle with the given stamp. """
        gimbal_msg = Vector3Stamped()
        gimbal_msg.header.stamp = stamp
        gimbal_msg.header.frame_id = "gimbal_angle"
        gimbal_msg.vector.x = float(self._gimbal_actual_pitch)
        self._gimbal_angle_publisher.publish(gimbal_msg)


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