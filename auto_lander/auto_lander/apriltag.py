import sys
import signal
import cv2
import numpy as np
import tf2_ros
import rclpy
import tf_transformations

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import TransformStamped
from mavros_msgs.srv import CommandLong
from pupil_apriltags import Detector as AprilTagDetector

from .vision_common import TagDefinition
from .vision_common import CameraIntrinsics
from .vision_common import OdometryBuffer
from .vision_common import FrameRecorder
from .vision_common import camera_pose_in_level_frame

""" AprilTag Landing Pad Detection Node.

    Runs the fast, precise perception loop: AprilTag detection -> solvePnP pose ->
    gimbal control -> base_link -> landing_pad_link TF broadcast. This TF broadcast IS
    the primary (full 6-DOF, best-conditioned) measurement stream the orchestrator's
    UKF consumes.

    Split out from the combined vision_perception node into its own process so this
    loop can run on its own executor thread/core, independent of the (much heavier,
    and intentionally slower) YOLO pipeline in yolo.py - a slow YOLO inference tick can
    no longer delay the timer driving the gimbal and TF broadcast here.

    This node subscribes to /camera/image_raw for its frames. In webcam mode it is
    also the one that owns the physical device and republishes raw frames onto
    /camera/image_raw so yolo.py can consume the exact same uniform topic regardless
    of image_source - two processes can't both open the same webcam device reliably.

    Because it's now a separate process from yolo.py, this node also publishes its
    commanded gimbal angle on /landing_pad/gimbal_angle: YOLO needs to know where the
    camera was actually pointed at its own frame's timestamp for its ground-plane
    back-projection, but no longer has direct access to this node's in-memory state.
"""


class AprilTagNode(Node):
    """ Defines the AprilTag perception node. """

    def __init__(self) -> None:
        """ Initialise the AprilTag perception node. """

        super().__init__("apriltag_landing_pad_node")

        # ---- NODE PARAMETERS ----
        self.declare_parameter("diagnostics_enabled", True)
        self.diagnostics_enabled = (
            self.get_parameter("diagnostics_enabled").get_parameter_value().bool_value
        )

        self.declare_parameter("enable_debug_publish", False)
        self.enable_debug_publish = (
            self.get_parameter("enable_debug_publish").get_parameter_value().bool_value
        )

        self.declare_parameter("image_source", "topic")
        self.image_source = (
            self.get_parameter("image_source").get_parameter_value().string_value
        )

        self.declare_parameter("webcam_index", 0)
        self.webcam_index = int(
            self.get_parameter("webcam_index").get_parameter_value().integer_value
        )

        self.declare_parameter("show_debug_window", False)
        self.show_debug_window = (
            self.get_parameter("show_debug_window").get_parameter_value().bool_value
        )

        self.declare_parameter("save_frames", False)
        self.save_frames = (
            self.get_parameter("save_frames").get_parameter_value().bool_value
        )

        self.declare_parameter("create_video", True)
        self.create_video = (
            self.get_parameter("create_video").get_parameter_value().bool_value
        )

        self.declare_parameter("video_fps", 30.0)
        self.video_fps = float(
            self.get_parameter("video_fps").get_parameter_value().double_value
        )

        self.declare_parameter("output_dir", "")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        self.declare_parameter("apriltag_processing_rate", 15.0)
        self._apriltag_processing_rate = float(
            self.get_parameter("apriltag_processing_rate").get_parameter_value().double_value
        )

        # Only used in webcam mode: how fast we pull fresh frames off the device.
        self.declare_parameter("frame_capture_rate", 30.0)
        self._frame_capture_rate = float(
            self.get_parameter("frame_capture_rate").get_parameter_value().double_value
        )

        self.declare_parameter("imgsz_width", 640)
        self._image_width = int(
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )

        self.declare_parameter("imgsz_height", 480)
        self._image_height = int(
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )

        self.get_logger().info(
            f"AprilTag processing rate: {self._apriltag_processing_rate} Hz"
        )
        self.get_logger().info(
            f"Video recording parameters: save_frames={self.save_frames}, "
            f"create_video={self.create_video}, video_fps={self.video_fps}"
        )

        # ---- CAMERA INTRINSICS ----
        self._intrinsics = CameraIntrinsics(self._image_width, self._image_height)
        self._camera_matrix = self._intrinsics.matrix
        self._dist_coeffs = self._intrinsics.dist_coeffs

        # ---- GIMBAL CONTROLLER PARAMETERS ----
        self._gimbal_Kp = 0.03                # deg output per deg error
        self._gimbal_Kd = 0.005
        self._gimbal_prev_error = 0.0
        self._gimbal_max_slew_deg_s = 60.0
        self._gimbal_last_cmd_time = None

        self._servo_angle = -90.0
        self._gimbal_servo_ID = 10

        self._servo_min_angle = -135.0
        self._servo_max_angle = 45.0

        self._servo_pwm_min = 1100
        self._servo_pwm_max = 1900

        # ---- APRILTAG PARAMETERS ----
        # IDs must be valid tag36h11 IDs (0-586).
        SPACING = 0.341
        MAIN = -0.0912 + 0.15  # Move forward, but actual is at -0.0912
        self._tags = {
            1: TagDefinition(size=0.481, position=(0.0, MAIN, 0.0)),
            2: TagDefinition(size=0.072, position=(0.0, MAIN + SPACING, 0.0)),
            3: TagDefinition(size=0.072, position=(0.0, MAIN - SPACING, 0.0)),
        }

        self.detector = AprilTagDetector(
            families="tag36h11",
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.75,
            debug=0,
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
        self._odom_buffer = OdometryBuffer(window_s=1.0)

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._webcam_publisher = self.create_publisher(Image, "/image", 10)
        self._landing_pad_found_publisher = self.create_publisher(
            Bool, "/landing_pad/found", 10
        )
        # Published every tick so yolo.py can time-align the gimbal angle with its
        # own frames (see docstring above / GimbalAngleBuffer in vision_common.py).
        self._gimbal_angle_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/gimbal_angle", 10
        )

        # ---- SERVICES ----
        self.client = self.create_client(CommandLong, "/mavros/cmd/command")

        # ---- TF2 ----
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- DIAGNOSTICS ----
        self._apriltag_pipeline_timing_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/pipeline_timing", 10
        )

        # ---- FRAME RECORDING ----
        self._frame_recorder = FrameRecorder(
            self.get_logger(), "apriltag", self.output_dir, self.video_fps,
            self.save_frames, self.create_video,
        )

        if self.show_debug_window:
            cv2.namedWindow("AprilTag Debug", cv2.WINDOW_AUTOSIZE)

        # ---- IMAGE SOURCE SETUP ----
        self._img_msg = None
        self._img_received_time = None
        self._img_seq = 0
        self._last_processed_seq = -1

        _img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if self.image_source == "topic":
            self.image_subscription = self.create_subscription(
                Image, "/camera/image_raw", self._image_store_callback, _img_qos
            )
            self.get_logger().info(
                "AprilTag (tag36h11) node started in TOPIC mode, waiting for "
                "MAVROS odometry and image topic..."
            )
        else:
            # Webcam mode: this node owns the physical device and republishes raw
            # frames on /camera/image_raw so yolo.py has a single, uniform topic to
            # subscribe to regardless of image_source.
            self._image_raw_publisher = self.create_publisher(
                Image, "/camera/image_raw", _img_qos
            )

            self.cap = None
            backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]

            for backend in backends_to_try:
                try:
                    self.cap = cv2.VideoCapture(self.webcam_index, backend)
                    if self.cap.isOpened():
                        self.get_logger().info(
                            f"Successfully opened camera {self.webcam_index} with backend {backend}"
                        )
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    self.get_logger().warning(
                        f"Failed to open camera with backend {backend}: {e}"
                    )
                    if self.cap:
                        self.cap.release()
                        self.cap = None

            if self.cap is None or not self.cap.isOpened():
                self.get_logger().error(
                    f"Could not open webcam at index {self.webcam_index}. "
                    f"Make sure your user is in the 'video' group: sudo usermod -a -G video $USER"
                )
            else:
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._image_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._image_height)

                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                self.get_logger().info(
                    f"AprilTag (tag36h11) node started in WEBCAM mode. Camera properties: "
                    f"FPS={actual_fps}, Width={actual_width}, Height={actual_height}"
                )
                self.get_logger().info(
                    "Webcam frames are being republished on /camera/image_raw for yolo.py."
                )

            self._frame_capture_timer = self.create_timer(
                1.0 / self._frame_capture_rate, self._frame_capture_timer_callback
            )

        # ---- PROCESSING TIMER ----
        self._apriltag_timer = self.create_timer(
            1.0 / self._apriltag_processing_rate, self._apriltag_timer_callback
        )

    # ---- RECEPTION CALLBACKS ----
    def _image_store_callback(self, msg) -> None:
        """ Store the latest incoming image message.

        :param msg: Incoming Image message from the camera topic
        """

        self._img_msg = msg
        self._img_received_time = self.get_clock().now()
        self._img_seq += 1

    def _frame_capture_timer_callback(self) -> None:
        """ Timer callback (webcam mode only): grabs a fresh frame off the device,
            stores it locally, and republishes it on /camera/image_raw.
        """

        if not (hasattr(self, "cap") and self.cap is not None and self.cap.isOpened()):
            self.get_logger().warning("Webcam not opened.")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read frame from webcam.")
            return

        frame = cv2.resize(
            frame,
            (self._image_width, self._image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()

        self._img_msg = msg
        self._img_received_time = self.get_clock().now()
        self._img_seq += 1

        self._image_raw_publisher.publish(msg)

    def _quad_odometry_callback(self, msg: Odometry) -> None:
        """ Obtain FCU Odometry, buffering stamped attitude for time-correct lookups.

        :param msg: Incoming Odometry message
        """

        self._odom_buffer.push(msg)

    def _get_new_frame(self):
        """ Pull the latest frame, but only if it's newer than the last frame this
            node already processed.

        :return: (frame, stamp) if a new frame is available, otherwise (None, None)
        """

        if self._img_msg is None or self._img_seq == self._last_processed_seq:
            return None, None

        self._last_processed_seq = self._img_seq

        msg = self._img_msg
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = msg.header.stamp

        if frame.shape[0] != self._image_height or frame.shape[1] != self._image_width:
            frame = cv2.resize(
                frame,
                (self._image_width, self._image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        return frame, stamp

    def _apriltag_timer_callback(self) -> None:
        """ Fires at apriltag_processing_rate. Detects AprilTags, estimates pose,
            drives the gimbal controller, and broadcasts the base_link ->
            landing_pad_link TF.
        """

        frame, stamp = self._get_new_frame()
        if frame is None:
            return  # no new frame since this node last ran

        odom = self._odom_buffer.get_at(stamp)
        if odom is None:
            self.get_logger().warn(
                "No odometry buffered yet - skipping frame", throttle_duration_sec=1.0
            )
            return
        q, _altitude = odom  # altitude unused here - solvePnP gives metric depth directly

        apriltag_detections = self._apriltag_detection(frame)

        landing_pad_found = len(apriltag_detections) > 0
        self._landing_pad_found_publisher.publish(Bool(data=landing_pad_found))

        if landing_pad_found:
            # Select the largest visible tag by apparent (pixel) area
            best = max(
                apriltag_detections,
                key=lambda d: cv2.contourArea(d.corners.astype(np.float32)),
            )
            tag_id = best.tag_id

            # Reorder pupil_apriltags' corners to match TagDefinition.object_points'
            # [top-left, top-right, bottom-right, bottom-left] convention
            image_points = best.corners[[1, 0, 3, 2]].astype(np.float32)
            tag = self._tags[tag_id]
            object_points = tag.object_points

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            if success:
                self._gimbal_controller(image_points)

                if self.show_debug_window:
                    cv2.drawFrameAxes(
                        frame,
                        self._camera_matrix,
                        self._dist_coeffs,
                        rvec,
                        tvec,
                        tag.size * 0.5,
                    )

                tf_base_to_pad = self._compose_base_to_landing_pad(
                    stamp, q, self._servo_angle, rvec, tvec, tag_id, tag.position
                )
                self._tf_broadcaster.sendTransform(tf_base_to_pad)
                # This TF broadcast IS the AprilTag measurement update the UKF
                # consumes (full 6-DOF: translation + rotation, from solvePnP).

        # Pipeline Latency Diagnostics
        if self.diagnostics_enabled:
            transform_ready_time = self.get_clock().now()
            timing_msg = Vector3Stamped()
            timing_msg.header.stamp = stamp  # t0 same as the TF's stamp, aligned
            timing_msg.header.frame_id = "pipeline_timing"
            timing_msg.vector.x = self._img_received_time.nanoseconds / 1e9  # t1 #type: ignore
            timing_msg.vector.y = transform_ready_time.nanoseconds / 1e9     # t2
            self._apriltag_pipeline_timing_publisher.publish(timing_msg)

        # Publish gimbal angle every tick (regardless of pose update success) so
        # yolo.py always has a fresh time-series to interpolate against.
        gimbal_msg = Vector3Stamped()
        gimbal_msg.header.stamp = self.get_clock().now().to_msg()
        gimbal_msg.header.frame_id = "gimbal_angle"
        gimbal_msg.vector.x = float(self._servo_angle)
        self._gimbal_angle_publisher.publish(gimbal_msg)

        self._gimbal_publisher(self._servo_angle)

        if self.show_debug_window:
            self._show_apriltag_debug(frame, apriltag_detections)
            cv2.imshow("AprilTag Debug", frame)
            cv2.waitKey(1)

        if self.save_frames or self.create_video:
            self._frame_recorder.save(frame)

        if self.enable_debug_publish:
            pub_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self._webcam_publisher.publish(pub_msg)

    def _apriltag_detection(self, frame):
        """ Run AprilTag detection and return recognised tag detections.

        :param frame: BGR image frame.
        :return: List of recognised AprilTag detections.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray_frame)  # type: ignore

        valid_detections = [d for d in detections if d.tag_id in self._tags]  # type: ignore
        return valid_detections

    def _show_apriltag_debug(self, frame, detections):
        """ Show apriltag debug.

        :param frame: Image frame
        :param detections: Apriltag detections list
        """
        for d in detections:  # type: ignore
            pts = d.corners.astype(np.int32)
            colour = (0, 255, 0) if d.tag_id in self._tags else (0, 165, 255)
            cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)
            cv2.putText(
                frame,
                str(d.tag_id),
                (int(d.center[0]), int(d.center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    def _send_servo_command(self, servo_id, pwm_value) -> None:
        """ Sends servo command via Mavlink.

        :param servo_id: Servo ID to actuate
        :param pwm_value: PWM value to command to servo
        """
        req = CommandLong.Request()
        req.command = 183
        req.param1 = float(servo_id)
        req.param2 = float(pwm_value)

        self.client.call_async(req)

    def _gimbal_controller(self, image_points) -> None:
        """ Determines gimbal required output to centre on tag.

        :param image_points: Tag corner points
        """

        now = self.get_clock().now()
        if self._gimbal_last_cmd_time is None:
            dt = 1.0 / self._apriltag_processing_rate
        else:
            dt = (now - self._gimbal_last_cmd_time).nanoseconds / 1e9
            dt = max(dt, 1e-3)
        self._gimbal_last_cmd_time = now

        centre_y = np.mean(image_points[:, 1])
        pixel_error = centre_y - self._image_height / 2
        fy = self._camera_matrix[1, 1]
        angle_error = np.degrees(np.arctan2(pixel_error, fy))

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

    def _gimbal_publisher(self, servo_angle) -> None:
        """ Publish the commanded gimbal angle to the gimbal as a PWM signal.

        :param servo_angle: Desired gimbal angle to servo
        """
        pwm = int(
            ((servo_angle - self._servo_min_angle) / 180.0)
            * (self._servo_pwm_max - self._servo_pwm_min)
            + self._servo_pwm_min
        )
        pwm = np.clip(pwm, self._servo_pwm_min, self._servo_pwm_max)

        self._send_servo_command(self._gimbal_servo_ID, pwm)

    @staticmethod
    def _compose_base_to_landing_pad(
        stamp, quad_rotation, servo_angle, rvec, tvec, tag_id, tag_position
    ):
        """ Compose quad_local->quad_body->cam->tag->landing_pad into a single
            base_link->landing_pad_link transform.

        :param stamp:         ROS timestamp
        :param quad_rotation: Quadcopter rotation quaternion
        :param servo_angle:   Current gimbal servo angle (degrees)
        :param rvec:          AprilTag rotation vector (camera->tag)
        :param tvec:          AprilTag translation vector (camera->tag)
        :param tag_id:        Detected tag ID (unused here, kept for clarity)
        :param tag_position:  (x, y, z) offset of this tag on the landing pad
        :return:              TransformStamped: base_link -> landing_pad_link
        """

        T_quad_cam = camera_pose_in_level_frame(quad_rotation, servo_angle)

        # Cam -> Tag
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = tvec.reshape(3)

        # Tag -> landing pad
        q_tag_pad = tf_transformations.quaternion_from_euler(0.0, 0.0, 1.5707963)
        T_tag_pad = tf_transformations.quaternion_matrix(q_tag_pad)
        T_tag_pad[:3, 3] = np.array(tag_position)

        T_local_pad = T_quad_cam @ T_cam_tag @ T_tag_pad

        t_out = T_local_pad[:3, 3]
        q_out = tf_transformations.quaternion_from_matrix(T_local_pad)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "local"
        tf_msg.child_frame_id = "landing_pad_link"
        tf_msg.transform.translation.x = float(t_out[0])
        tf_msg.transform.translation.y = float(t_out[1])
        tf_msg.transform.translation.z = float(t_out[2])
        tf_msg.transform.rotation.x = float(q_out[0])
        tf_msg.transform.rotation.y = float(q_out[1])
        tf_msg.transform.rotation.z = float(q_out[2])
        tf_msg.transform.rotation.w = float(q_out[3])

        return tf_msg


# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = AprilTagNode()

    shutdown_in_progress = False

    def signal_handler(signum, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            node.get_logger().warn(
                "Second interrupt received! Force terminating without video creation..."
            )
            sys.exit(1)
        else:
            shutdown_in_progress = True
            node.get_logger().info(
                "Interrupt received, creating video before shutdown (press Ctrl+C again to force quit)..."
            )
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally:
        try:
            node._frame_recorder.finalize()
        except Exception as e:
            node.get_logger().error(f"Failed to create video during shutdown: {e}")

        if hasattr(node, "cap") and node.cap is not None:
            node.cap.release()
        node.destroy_node()
        if getattr(node, "show_debug_window", True):
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
