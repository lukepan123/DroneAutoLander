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

from pupil_apriltags import Detector as AprilTagDetector

from .vision_common import TagDefinition
from .vision_common import CameraIntrinsics
from .vision_common import OdometryBuffer
from .vision_common import GimbalAngleBuffer
from .vision_common import FrameRecorder
from .vision_common import camera_pose_in_level_frame
from .vision_common import image_target_message
from .vision_common import load_camera_mount
from .vision_common import stamp_to_sec

""" AprilTag Landing Pad Detection Node.

    Runs the fast, precise perception loop: AprilTag detection -> solvePnP pose ->
    landing-pad measurement. This node does not own the gimbal actuator - it publishes
    its detected target point on the image to gimbal_controller.py, which independently
    controls the servo.

    Split out from the combined vision_perception node into its own process so this
    loop can run on its own executor thread/core, independent of the (much heavier,
    and intentionally slower) YOLO pipeline in yolo.py - a slow YOLO inference tick can
    no longer delay the AprilTag detector here.

    This node subscribes to /camera/image_raw for its frames. In webcam mode it is
    also the one that owns the physical device and republishes raw frames onto
    /camera/image_raw so yolo.py can consume the exact same uniform topic regardless
    of image_source - two processes can't both open the same webcam device reliably.

    The camera pose is built locally rather than read from TF: buffered FCU odometry
    (attitude) and the gimbal angle from /landing_pad/gimbal_angle are both
    interpolated to the image timestamp and fed through camera_pose_in_level_frame.
    That pose is composed with the solvePnP camera -> tag transform and the tag ->
    landing pad transform to produce the full local -> landing_pad_link measurement
    used by the UKF. There is no camera TF lookup, so a late-arriving sample can't
    cause a TF extrapolation error or drop the frame.
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

        # ---- WEBCAM PARAMETERS ----
        self.declare_parameter("webcam_index", 0)
        self.webcam_index = (
            self.get_parameter("webcam_index").get_parameter_value().integer_value
        )

        self.declare_parameter("webcam_fps", 30.0)
        self._webcam_fps = (
            self.get_parameter("webcam_fps").get_parameter_value().double_value
        )

        self.declare_parameter("show_debug_window", False)
        self.show_debug_window = (
            self.get_parameter("show_debug_window").get_parameter_value().bool_value
        )

        # ---- RECORDING PARAMETERS ----
        self.declare_parameter("save_frames", False)
        self.save_frames = (
            self.get_parameter("save_frames").get_parameter_value().bool_value
        )

        self.declare_parameter("create_video", True)
        self.create_video = (
            self.get_parameter("create_video").get_parameter_value().bool_value
        )

        self.declare_parameter("video_fps", 10.0)
        self.video_fps = (
            self.get_parameter("video_fps").get_parameter_value().double_value
        )

        self.declare_parameter("output_dir", "")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        # ---- PROCESSING PARAMETERS ----
        self.declare_parameter("apriltag_processing_rate", 10.0)
        self._apriltag_processing_rate = (
            self.get_parameter("apriltag_processing_rate")
            .get_parameter_value()
            .double_value
        )

        self.declare_parameter("frame_capture_rate", 24.0)
        self._frame_capture_rate = (
            self.get_parameter("frame_capture_rate")
            .get_parameter_value()
            .double_value
        )

        self.declare_parameter("img_width", 960)
        self._image_width = (
            self.get_parameter("img_width").get_parameter_value().integer_value
        )

        self.declare_parameter("img_height", 540)
        self._image_height = (
            self.get_parameter("img_height").get_parameter_value().integer_value
        )

        # ---- APRILTAG PARAMETERS ----
        self.declare_parameter("apriltag_family", "tag36h11")
        self.declare_parameter("apriltag_quad_decimate", 2.0)
        self.declare_parameter("apriltag_quad_sigma", 0.0)
        self.declare_parameter("apriltag_refine_edges", 1)
        self.declare_parameter("apriltag_decode_sharpening", 0.75)
        self.declare_parameter("apriltag_debug", 0)
        self.declare_parameter("apriltag_tag_ids", [11, 21, 31])
        self.declare_parameter("apriltag_spacing_m", 0.341)
        self.declare_parameter("apriltag_main_offset_m", -0.0912 + 0.15)
        self.declare_parameter("apriltag_main_tag_size_m", 0.481)
        self.declare_parameter("apriltag_small_tag_size_m", 0.072)
        self.declare_parameter("apriltag_tag_to_pad_yaw_deg", 90.0)
        self._apriltag_family = (
            self.get_parameter("apriltag_family").get_parameter_value().string_value
        )
        self._apriltag_quad_decimate = (
            self.get_parameter("apriltag_quad_decimate")
            .get_parameter_value()
            .double_value
        )
        self._apriltag_quad_sigma = (
            self.get_parameter("apriltag_quad_sigma")
            .get_parameter_value()
            .double_value
        )
        self._apriltag_refine_edges = (
            self.get_parameter("apriltag_refine_edges")
            .get_parameter_value()
            .integer_value
        )
        self._apriltag_decode_sharpening = (
            self.get_parameter("apriltag_decode_sharpening")
            .get_parameter_value()
            .double_value
        )
        self._apriltag_debug = (
            self.get_parameter("apriltag_debug").get_parameter_value().integer_value
        )
        tag_ids = [
            int(v)
            for v in self.get_parameter("apriltag_tag_ids")
            .get_parameter_value()
            .integer_array_value
        ]
        self._apriltag_spacing_m = (
            self.get_parameter("apriltag_spacing_m").get_parameter_value().double_value
        )
        self._apriltag_main_offset_m = (
            self.get_parameter("apriltag_main_offset_m")
            .get_parameter_value()
            .double_value
        )
        self._apriltag_main_tag_size_m = (
            self.get_parameter("apriltag_main_tag_size_m")
            .get_parameter_value()
            .double_value
        )
        self._apriltag_small_tag_size_m = (
            self.get_parameter("apriltag_small_tag_size_m")
            .get_parameter_value()
            .double_value
        )
        self._apriltag_tag_to_pad_yaw_rad = np.deg2rad(
            self.get_parameter("apriltag_tag_to_pad_yaw_deg")
            .get_parameter_value()
            .double_value
        )
        if len(tag_ids) != 3:
            raise ValueError("apriltag_tag_ids must contain exactly three IDs")

        # ---- CAMERA PARAMETERS ----
        self.declare_parameter("camera_fov_horizontal", 2.7925268)
        self._camera_fov_horizontal = (
            self.get_parameter("camera_fov_horizontal")
            .get_parameter_value()
            .double_value
        )

        # Camera mount offsets (shared defaults in vision_common.load_camera_mount)
        self._camera_mount = load_camera_mount(self)

        # How much odometry / gimbal angle history to keep for image-time alignment
        self.declare_parameter("odometry_buffer_window_s", 1.0)
        self._odometry_buffer_window_s = (
            self.get_parameter("odometry_buffer_window_s")
            .get_parameter_value()
            .double_value
        )

        self.declare_parameter("camera_distortion_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        camera_distortion_coeffs = (
            self.get_parameter("camera_distortion_coeffs")
            .get_parameter_value()
            .double_array_value
        )
        if len(camera_distortion_coeffs) != 5:
            raise ValueError("camera_distortion_coeffs must contain exactly five values")

        self.get_logger().info(
            f"AprilTag processing rate: {self._apriltag_processing_rate} Hz"
        )
        self.get_logger().info(
            f"Video recording parameters: save_frames={self.save_frames}, "
            f"create_video={self.create_video}, video_fps={self.video_fps}"
        )
        # ---- CAMERA INTRINSICS ----
        self._intrinsics = CameraIntrinsics(
            self._image_width,
            self._image_height,
            self._camera_fov_horizontal,
            tuple(camera_distortion_coeffs), # type: ignore
        )
        self._camera_matrix = self._intrinsics.matrix
        self._dist_coeffs = self._intrinsics.dist_coeffs_array

        # ---- APRILTAG DEFINITIONS ----
        # IDs must be valid tag36h11 IDs (0-586).
        SPACING = self._apriltag_spacing_m
        MAIN = self._apriltag_main_offset_m
        self._tags = {
            tag_ids[0]: TagDefinition(
                size=self._apriltag_main_tag_size_m,
                position=(0.0, MAIN, 0.0),
            ),
            tag_ids[1]: TagDefinition(
                size=self._apriltag_small_tag_size_m,
                position=(0.0, MAIN + SPACING, 0.0),
            ),
            tag_ids[2]: TagDefinition(
                size=self._apriltag_small_tag_size_m,
                position=(0.0, MAIN - SPACING, 0.0),
            ),
        }

        self.detector = AprilTagDetector(
            families=self._apriltag_family,
            quad_decimate=self._apriltag_quad_decimate,
            quad_sigma=self._apriltag_quad_sigma,
            refine_edges=self._apriltag_refine_edges,
            decode_sharpening=self._apriltag_decode_sharpening,
            debug=self._apriltag_debug,
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
        self._odom_buffer = OdometryBuffer(window_s=self._odometry_buffer_window_s)

        # Gimbal angle (deg) from gimbal_controller.py, interpolated to image stamps.
        self._gimbal_buffer = GimbalAngleBuffer(
            window_s=self._odometry_buffer_window_s
        )
        self._gimbal_angle_sub = self.create_subscription(
            Vector3Stamped,
            "/landing_pad/gimbal_angle",
            self._gimbal_angle_callback,
            10,
        )

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
                "image topic, odometry and gimbal angle..."
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
                    "Make sure your user is in the 'video' group: sudo usermod -a -G video $USER"
                )
            else:
                self.cap.set(cv2.CAP_PROP_FPS, self._webcam_fps)
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

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._webcam_publisher = self.create_publisher(Image, "/image", 10)
        self._landing_pad_found_publisher = self.create_publisher(
            Bool, "/landing_pad/found", 10
        )
        self._apriltag_target_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/apriltag_target", 10
        )

        # ---- TF2 ----
        # Broadcast only: this node publishes the landing pad measurement as TF but
        # no longer listens to /tf.
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- DIAGNOSTICS ----
        self._apriltag_pipeline_timing_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/pipeline_timing", 10
        )

        # ---- FRAME RECORDING ----
        self._frame_recorder = FrameRecorder(
            self.get_logger(),
            "apriltag",
            self.output_dir,
            self.video_fps,
            self.save_frames,
            self.create_video,
        )

        if self.show_debug_window:
            cv2.namedWindow("AprilTag Debug", cv2.WINDOW_AUTOSIZE)

        # ---- IMAGE BUFFER ----
        self._img_msg = None
        self._img_received_time = None
        self._img_seq = 0
        self._last_processed_seq = -1

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

    def _quad_odometry_callback(self, msg: Odometry) -> None:
        """ Buffer stamped FCU attitude/altitude for image-timestamp alignment.

        :param msg: Incoming Odometry message
        """
        self._odom_buffer.push(msg)

    def _gimbal_angle_callback(self, msg: Vector3Stamped) -> None:
        """ Buffer the measured gimbal pitch for image-timestamp alignment.

        :param msg: Vector3Stamped from gimbal_controller.py; vector.x is the actual
                    gimbal pitch in degrees
        """
        self._gimbal_buffer.push(stamp_to_sec(msg.header.stamp), msg.vector.x)

    def _camera_pose_at(self, stamp):
        """ Camera pose in the local-level frame at an image timestamp, built from the
            buffered odometry attitude and gimbal angle (both interpolated to `stamp`).

        :param stamp: builtin_interfaces/Time, the image's header.stamp
        :return: 4x4 camera pose (level-frame <- camera-frame), or None if either
                 buffer has no data yet
        """
        odom = self._odom_buffer.get_at(stamp)
        gimbal_deg = self._gimbal_buffer.get_at(stamp)
        if odom is None or gimbal_deg is None:
            self.get_logger().warn(
                "No odometry/gimbal angle buffered yet - skipping pose",
                throttle_duration_sec=1.0,
            )
            return None
        q, _altitude = odom
        return camera_pose_in_level_frame(q, gimbal_deg, self._camera_mount)

    def _apriltag_timer_callback(self) -> None:
        """ Fires at apriltag_processing_rate. Detects AprilTags, publishes the image
            target for the independent gimbal controller, and broadcasts the local ->
            landing_pad_link measurement when solvePnP and the camera pose are available.
        """

        frame, stamp = self._get_new_frame()
        if frame is None:
            return  # no new frame since this node last ran

        apriltag_detections = self._apriltag_detection(frame)
        landing_pad_found = False

        if len(apriltag_detections) > 0:
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

            # Publish the detected image point immediately so the independent gimbal
            # controller can track the target even if solvePnP later fails.
            centre_x = np.mean(image_points[:, 0])
            centre_y = np.mean(image_points[:, 1])
            self._apriltag_target_publisher.publish(
                image_target_message(
                    stamp, centre_x, centre_y, self._intrinsics, "apriltag_target"
                )
            )

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            T_local_cam = self._camera_pose_at(stamp) if success else None

            if T_local_cam is not None:
                landing_pad_found = True

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
                    stamp,
                    T_local_cam,
                    rvec,
                    tvec,
                    tag_id,
                    tag.position,
                )
                self._tf_broadcaster.sendTransform(tf_base_to_pad)
                # This TF broadcast IS the AprilTag measurement update the UKF
                # consumes (full 6-DOF: translation + rotation, from solvePnP).

        self._landing_pad_found_publisher.publish(Bool(data=landing_pad_found))

        # Pipeline Latency Diagnostics
        if self.diagnostics_enabled:
            transform_ready_time = self.get_clock().now()
            timing_msg = Vector3Stamped()
            timing_msg.header.stamp = stamp  # t0 same as the TF's stamp, aligned
            timing_msg.header.frame_id = "pipeline_timing"
            timing_msg.vector.x = self._img_received_time.nanoseconds / 1e9  # t1  #type: ignore
            timing_msg.vector.y = transform_ready_time.nanoseconds / 1e9     # t2
            self._apriltag_pipeline_timing_publisher.publish(timing_msg)

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
        detections = self.detector.detect(gray_frame)  #type: ignore

        valid_detections = [d for d in detections if d.tag_id in self._tags]  #type: ignore
        return valid_detections

    def _show_apriltag_debug(self, frame, detections):
        """ Show apriltag debug.

        :param frame: Image frame
        :param detections: Apriltag detections list
        """
        for d in detections:  #type: ignore
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


    def _compose_base_to_landing_pad(
        self, stamp, T_local_cam, rvec, tvec, tag_id, tag_position
    ):
        """ Compose quad_local->quad_body->cam->tag->landing_pad into a single
            base_link->landing_pad_link transform.

        :param stamp:       ROS timestamp
        :param T_local_cam: Camera pose in the local-level frame at the image timestamp
                            (camera_pose_in_level_frame)
        :param rvec:         AprilTag rotation vector (camera->tag)
        :param tvec:         AprilTag translation vector (camera->tag)
        :param tag_id:       Detected tag ID (unused here, kept for clarity)
        :param tag_position:  (x, y, z) offset of this tag on the landing pad
        :return:              TransformStamped: base_link -> landing_pad_link
        """

        # Cam -> Tag
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = tvec.reshape(3)

        # Tag -> landing pad
        q_tag_pad = tf_transformations.quaternion_from_euler(
            0.0, 0.0, self._apriltag_tag_to_pad_yaw_rad
        )
        T_tag_pad = tf_transformations.quaternion_matrix(q_tag_pad)
        T_tag_pad[:3, 3] = np.array(tag_position)

        T_local_pad = T_local_cam @ T_cam_tag @ T_tag_pad

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
