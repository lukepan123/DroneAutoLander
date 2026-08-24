import os
import sys
import signal
import cv2
import numpy as np
import numpy.typing as npt
import tf2_ros
import rclpy
import tf_transformations
import bisect

from cv_bridge import CvBridge
from collections import deque
from dataclasses import dataclass
from dataclasses import field
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
from datetime import datetime
from typing import cast

from ultralytics import YOLO
from ultralytics import utils as yutils
import logging
import warnings

""" Landing Pad Detection Node handles all high level processing of the camera feed and 
    produces a transformation from the quadcopter to the landing pad for the controller 
    to utilise.

    AprilTag and YOLO run on two independent timers so each pipeline can be tuned/paced
    separately: AprilTag is the fast, precise loop that drives the gimbal and TF broadcast;
    YOLO is a coarser, slower loop intended to eventually feed its own measurement update
    into the UKF. Both pipelines pull from the same shared latest-frame buffer, gated by a
    monotonically increasing frame sequence number so neither pipeline reprocesses a frame
    the other already consumed, and neither blocks the other from seeing a new one.
"""

# ---- Ultralytics/OpenVINO runtime safety ----
os.environ["AUTOINSTALL"] = "0"
os.environ["YOLOv5_AUTOINSTALL"] = "0"
os.environ["OV_CPU_THREADS_NUM"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

yutils.ONLINE = False
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")


def _get_workspace_root() -> str | None:
    """ Find the workspace root by looking for colcon workspace structure. """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != "/":
        if (
            os.path.exists(os.path.join(current_dir, "src"))
            and os.path.exists(os.path.join(current_dir, "build"))
            and os.path.exists(os.path.join(current_dir, "install"))
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


@dataclass
class TagDefinition:
    """ Defines an AprilTag definition (tag size and position on the landing pad)."""

    size: float
    position: tuple[float, float, float]
    object_points: np.ndarray = field(init=False)

    def __post_init__(self):
        half = self.size / 2.0

        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )


class VisionPerception(Node):
    """ Defines the vision perception node.
    """

    def __init__(self) -> None:
        """ Initialise the vision perception node
        """

        super().__init__("landing_pad_detection_node")
        # ---- NODE PARAMETERS ----
        # If in sim, enable logging and certain diagnostics
        self.declare_parameter("diagnostics_enabled", True)
        self.diagnostics_enabled = (
            self.get_parameter("diagnostics_enabled").get_parameter_value().bool_value
        )

        # Publish the image topic from the camera after processing
        self.declare_parameter("enable_debug_publish", False)
        self.enable_debug_publish = (
            self.get_parameter("enable_debug_publish").get_parameter_value().bool_value
        )

        # Choose between camera image topic or direct webcam source
        self.declare_parameter("image_source", "topic")
        self.image_source = (
            self.get_parameter("image_source").get_parameter_value().string_value
        )

        # Webcam index number
        self.declare_parameter("webcam_index", 0)
        self.webcam_index = int(
            self.get_parameter("webcam_index").get_parameter_value().integer_value
        )

        # Show the camera viewport
        self.declare_parameter("show_debug_window", False)
        self.show_debug_window = (
            self.get_parameter("show_debug_window").get_parameter_value().bool_value
        )

        # Save individual frames from vision
        self.declare_parameter("save_frames", False)
        self.save_frames = (
            self.get_parameter("save_frames").get_parameter_value().bool_value
        )

        # Create video of vision throughout running of program
        self.declare_parameter("create_video", True)
        self.create_video = (
            self.get_parameter("create_video").get_parameter_value().bool_value
        )

        # Declare video FPS
        self.declare_parameter("video_fps", 30.0)
        self.video_fps = float(
            self.get_parameter("video_fps").get_parameter_value().double_value
        )

        # Declare video/frames output directory location
        self.declare_parameter("output_dir", "")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        # ---- PROCESSING RATES ----
        # AprilTag is the fast, precise loop that drives the gimbal + TF broadcast.
        self.declare_parameter("apriltag_processing_rate", 15.0)
        self._apriltag_processing_rate = float(
            self.get_parameter("apriltag_processing_rate").get_parameter_value().double_value
        )

        # YOLO is a coarser, heavier loop - intentionally slower by default.
        self.declare_parameter("yolo_processing_rate", 15.0)
        self._yolo_processing_rate = float(
            self.get_parameter("yolo_processing_rate").get_parameter_value().double_value
        )

        # Only used in webcam mode: how fast we pull fresh frames off the device,
        # independent of how fast either processing pipeline consumes them.
        self.declare_parameter("frame_capture_rate", 30.0)
        self._frame_capture_rate = float(
            self.get_parameter("frame_capture_rate").get_parameter_value().double_value
        )

        # Log parameter values for debugging
        self.get_logger().info(
            f"Video recording parameters: save_frames={self.save_frames}, create_video={self.create_video}, video_fps={self.video_fps}"
        )
        self.get_logger().info(
            f"Output directory: '{self.output_dir}' (empty means workspace root)"
        )
        self.get_logger().info(
            f"AprilTag processing rate: {self._apriltag_processing_rate} Hz, "
            f"YOLO processing rate: {self._yolo_processing_rate} Hz"
        )

        # ---- CAMERA PARAMETERS ----
        # Must match the IR input size you exported (default IRIS is 640 x 480)
        self.declare_parameter("imgsz_width", 640)
        self._image_width = int(
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )

        self.declare_parameter("imgsz_height", 480)
        self._image_height = int(
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )

        self._camera_fov_horizontal = 2.0  # radians (≈114.6°) – tune for your camera
        self._camera_fov_vertical = 2 * np.arctan(
            np.tan(self._camera_fov_horizontal / 2)
            / (self._image_width / self._image_height)
        )

        # Generate the camera matrix
        fx = self._image_width / (2 * np.tan(self._camera_fov_horizontal / 2))
        fy = self._image_height / (2 * np.tan(self._camera_fov_vertical / 2))
        self._camera_matrix = np.array(
            [
                [fx, 0, self._image_width / 2],
                [0, fy, self._image_height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        self._dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)

        # Used to back-project YOLO pixel detections into camera-frame rays (no known
        # object size for YOLO, so we can't solvePnP - see _estimate_yolo_ground_position)
        self._camera_matrix_inv = np.linalg.inv(self._camera_matrix)

        # ---- GIMBAL CONTROLLER PARAMETERS ----
        self._gimbal_Kp = 0.03                # now deg output per deg error
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
        MAIN = -0.0912 + 0.15 # Move forward, but actual is at -0.0912
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

        # ---- YOLO VISION PARAMETERS ----
        workspace_root = _get_workspace_root()

        default_yolo_model = (
            os.path.join(workspace_root, "ugv_yolo11n_openvino_model")
            if workspace_root
            else "ugv_yolo11n_openvino_model"
        )

        self.declare_parameter("yolo_enabled", True)
        self.declare_parameter("yolo_model_path", default_yolo_model)
        self.declare_parameter("yolo_conf_threshold", 0.5)

        self.yolo_enabled = (
            self.get_parameter("yolo_enabled")
            .get_parameter_value()
            .bool_value
        )

        self.yolo_model_path = (
            self.get_parameter("yolo_model_path")
            .get_parameter_value()
            .string_value
        )

        self.yolo_conf_threshold = float(
            self.get_parameter("yolo_conf_threshold")
            .get_parameter_value()
            .double_value
        )

        self.yolo_model = None
        if self.yolo_enabled:
            try:
                self.get_logger().info(
                    f"Loading YOLO model: {self.yolo_model_path}"
                )
                self.yolo_model = YOLO(self.yolo_model_path)
                self.get_logger().info(
                    f"YOLO model loaded successfully. Classes: "
                    f"{self.yolo_model.names}"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to load YOLO model '{self.yolo_model_path}': {e}"
                )
                raise

        # ---- SUBSCRIPTIONS ----
        _quad_odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._quad_odometry_sub = self.create_subscription(
            Odometry,
            "/mavros/global_position/local",
            self._quad_odometry_callback,
            _quad_odom_qos,
        )
        self.quad_odometry = Odometry()

        # Create odometry buffer for transformations in the past
        self.quad_odom_buffer: deque[tuple[float, list[float], float]] = deque(maxlen=400)
        self.quad_odom_buffer_window_s = 1.0  # sec, keep history to bracket image stamp

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._webcam_publisher = self.create_publisher(Image, "/image", 10)
        self._landing_pad_found_publisher = self.create_publisher(
            Bool, "/landing_pad/found", 10
        )

        # ---- SERVICES ----
        self.client = self.create_client(CommandLong, "/mavros/cmd/command")

        # ---- TF2 ----
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- DIAGNOSTICS AND LOGGING ----
        self._apriltag_pipeline_timing_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/pipeline_timing", 10
        )
        self._yolo_pipeline_timing_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/yolo_pipeline_timing", 10
        )

        # ---- INITIALISATION ----
        self._frame_count = 0
        self._saved_frames = []

        if self.save_frames or self.create_video:
            self._start_video_creation()
        else:
            self.get_logger().info("Frame saving DISABLED - no video will be created")

        if self.show_debug_window:
            cv2.namedWindow("AprilTag Debug", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("YOLO Debug", cv2.WINDOW_AUTOSIZE)

        # ---- IMAGE SOURCE SETUP ----
        # Shared latest-frame buffer. Both the AprilTag and YOLO timers read from this;
        # neither one clears it, since the other pipeline may not have consumed it yet.
        # `_frame_seq` is bumped every time a new frame arrives so each pipeline can tell
        # whether the buffer holds something it hasn't already processed.
        self._img_msg = None
        self._frame_seq = 0
        self._img_seq = 0
        self._last_apriltag_seq = -1
        self._last_yolo_seq = -1

        # Drop old frames instead of queuing
        _img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Create subscription based on image source
        if self.image_source == "topic":
            # Subscribe and store latest frame — processing happens in the timers
            self.image_subscription = self.create_subscription(
                Image, "/camera/image_raw", self._image_store_callback, _img_qos
            )
            self.get_logger().info(
                "AprilTag (tag36h11) detection node started in TOPIC mode, waiting for MAVROS altitude and image topic..."
            )
        else:
            # Open webcam
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
                    f"AprilTag (tag36h11) detection node started in WEBCAM mode. Camera properties: "
                    f"FPS={actual_fps}, Width={actual_width}, Height={actual_height}"
                )

            # Webcam frame acquisition gets its own timer, decoupled from both
            # processing pipelines - neither AprilTag nor YOLO should be responsible
            # for pulling frames off the device.
            self._frame_capture_timer = self.create_timer(
                1.0 / self._frame_capture_rate, self._frame_capture_timer_callback
            )

        # ---- PROCESSING TIMERS ----
        self._apriltag_timer = self.create_timer(
            1.0 / self._apriltag_processing_rate, self._apriltag_timer_callback
        )
        self._yolo_timer = self.create_timer(
            1.0 / self._yolo_processing_rate, self._yolo_timer_callback
        )

    # ---- RECEPTION CALLBACKS ----
    def _image_store_callback(self, msg) -> None:
        """ Store the latest incoming image message.

        :param msg: Incoming Image message from the camera topic
        """

        self._img_msg = msg
        self._img_received_time = self.get_clock().now()
        self._frame_seq += 1
        self._img_seq = self._frame_seq


    def _webcam_store_callback(self) -> None:
        """ Read one frame from the webcam and store it as a ROS Image message.
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
        self._frame_seq += 1
        self._img_seq = self._frame_seq


    def _frame_capture_timer_callback(self) -> None:
        """ Timer callback (webcam mode only): grabs a fresh frame off the device
            and stores it in the shared buffer for the processing timers to consume.
        """

        self._webcam_store_callback()


    def _quad_odometry_callback(self, msg: Odometry) -> None:
        """ Obtain FCU Odometry, buffering stamped attitude for time-correct lookups.
        
        :param msg: Incoming Image message from the odometry topic
        """

        # Keep for anything that just wants "latest"
        self.quad_odometry = msg

        # Otherwise push and timestamp into odometry buffer
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        altitude = msg.pose.pose.position.z  # buffered so YOLO's ground-plane estimate
                                              # can use altitude from the image's own timestamp
        self.quad_odom_buffer.append((t, q, altitude))

        cutoff = t - self.quad_odom_buffer_window_s
        while self.quad_odom_buffer and self.quad_odom_buffer[0][0] < cutoff:
            self.quad_odom_buffer.popleft()


    def _get_odometry_at(self, stamp) -> tuple[list[float], float] | None:
        """ Interpolate the quad's orientation (SLERP) and altitude (linear) to a
            specific timestamp. Altitude is needed by the YOLO ground-plane position
            estimate; AprilTag's pose comes from solvePnP and doesn't need it, but both
            pipelines pull orientation from here so they stay time-consistent.

        :param stamp: builtin_interfaces/Time, e.g. the image's header.stamp
        :return: ([x, y, z, w] quaternion, altitude) at that instant, or None if the
            buffer is empty
        """

        # If the buffer is empty, skip
        if not self.quad_odom_buffer:
            return None

        t_query = stamp.sec + stamp.nanosec * 1e-9
        times = [t for t, _, _ in self.quad_odom_buffer]

        # Clamp to oldest sample if time older than buffer
        if t_query <= times[0]:
            _, q, alt = self.quad_odom_buffer[0]
            return list(q), alt

        # Clamp to newest sample if time newer than buffer
        if t_query >= times[-1]:
            _, q, alt = self.quad_odom_buffer[-1]
            return list(q), alt

        idx = bisect.bisect_right(times, t_query)
        t0, q0, alt0 = self.quad_odom_buffer[idx - 1]
        t1, q1, alt1 = self.quad_odom_buffer[idx]

        if t1 <= t0:
            return list(q0), alt0

        fraction = (t_query - t0) / (t1 - t0)
        slerp_result = cast(
            npt.NDArray[np.floating], 
            tf_transformations.quaternion_slerp(q0, q1, fraction)
        )
        altitude = alt0 + fraction * (alt1 - alt0)
        return list(slerp_result), altitude


    def _get_new_frame(self, last_seq_attr: str):
        """ Pull the latest frame from the shared buffer, but only if it's newer than
            the last frame this particular pipeline processed. Each pipeline (AprilTag,
            YOLO) tracks its own "last processed" sequence number via `last_seq_attr`,
            so both can independently consume from the same buffer without stepping on
            each other or reprocessing a frame twice.

        :param last_seq_attr: name of the instance attribute holding this pipeline's
            last-processed frame sequence number (e.g. "_last_apriltag_seq")
        :return: (frame, stamp) if a new frame is available, otherwise (None, None)
        """

        if self._img_msg is None:
            return None, None

        if self._img_seq == getattr(self, last_seq_attr):
            return None, None  # already processed this frame

        setattr(self, last_seq_attr, self._img_seq)

        msg = self._img_msg
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = msg.header.stamp

        # Ensure inference size matches IR (handles topic frames of any size)
        if frame.shape[0] != self._image_height or frame.shape[1] != self._image_width:
            frame = cv2.resize(
                frame,
                (self._image_width, self._image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        return frame, stamp


    def _apriltag_timer_callback(self) -> None:
        """ Fires at apriltag_processing_rate. Detects AprilTags, estimates pose,
            drives the gimbal controller, and broadcasts the base_link -> landing_pad_link
            TF. Runs independently of the YOLO pipeline.
        """

        frame, stamp = self._get_new_frame("_last_apriltag_seq")
        if frame is None:
            return  # no new frame since this pipeline last ran

        odom = self._get_odometry_at(stamp)
        if odom is None:
            self.get_logger().warn("No odometry buffered yet — skipping frame",
                                   throttle_duration_sec=1.0)
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
                # Run gimbal controller
                self._gimbal_controller(image_points)

                # Draw pose axes for debugging
                if self.show_debug_window:
                    cv2.drawFrameAxes(
                        frame,
                        self._camera_matrix,
                        self._dist_coeffs,
                        rvec,
                        tvec,
                        tag.size * 0.5,
                    )

                # Broadcast landing_pad position relative to camera frame
                tf_base_to_pad = self._compose_base_to_landing_pad(
                    stamp, q, self._servo_angle, rvec, tvec, tag_id, tag.position
                )
                self._tf_broadcaster.sendTransform(tf_base_to_pad)
                # This TF broadcast IS the AprilTag measurement update the UKF consumes
                # (full 6-DOF: translation + rotation, from solvePnP).

        # Pipeline Latency Diagnostics
        if self.diagnostics_enabled:
            transform_ready_time = self.get_clock().now()
            timing_msg = Vector3Stamped()
            timing_msg.header.stamp = stamp # t0 same as the TF's stamp, aligned
            timing_msg.header.frame_id = "pipeline_timing"
            timing_msg.vector.x = self._img_received_time.nanoseconds / 1e9 # t1
            timing_msg.vector.y = transform_ready_time.nanoseconds / 1e9    # t2
            self._apriltag_pipeline_timing_publisher.publish(timing_msg)

        # Publish gimbal angle regardless of pose update success
        self._gimbal_publisher(self._servo_angle)

        # Show the output image after AprilTag detection (if debug window enabled)
        if self.show_debug_window:
            self._show_apriltag_debug(frame, apriltag_detections)
            cv2.imshow("AprilTag Debug", frame)
            cv2.waitKey(1)

        # Save frame if enabled
        if self.save_frames or self.create_video:
            self._save_video_from_frame(frame)

        if self.enable_debug_publish:
            pub_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self._webcam_publisher.publish(pub_msg)


    def _yolo_timer_callback(self) -> None:
        """ Fires at yolo_processing_rate. Runs the coarse YOLO landing-pad detector
            independently of the AprilTag pipeline. Intended to eventually publish its
            own measurement update to the UKF.
        """

        if not self.yolo_enabled or self.yolo_model is None:
            return

        frame, stamp = self._get_new_frame("_last_yolo_seq")
        if frame is None:
            return  # no new frame since this pipeline last ran

        odom = self._get_odometry_at(stamp)
        if odom is None:
            self.get_logger().warn("No odometry buffered yet — skipping frame",
                                   throttle_duration_sec=1.0)
            return
        q, altitude = odom

        detection = self._yolo_detection(frame)

        translation = None
        if detection is not None:
            cx, cy, bw, bh, confidence = detection

            # If we have a detection, then the landing pad is found
            self._landing_pad_found_publisher.publish(Bool(data=True))

            # A single 2D box gives a bearing to the pad, not depth or orientation, so
            # we recover (x, y, z) via a flat-ground assumption: cast the ray through the
            # box centre and intersect it with the ground plane, using altitude to fix
            # the scale. This carries no rotation information - see
            # _compose_base_to_landing_pad_yolo, which publishes identity rotation.
            translation = self._estimate_yolo_ground_position(
                cx, cy, q, self._servo_angle, altitude
            )

            if translation is not None:
                tf_base_to_pad_yolo = self._compose_base_to_landing_pad_yolo(
                    stamp, translation
                )
                self._tf_broadcaster.sendTransform(tf_base_to_pad_yolo)
                # This TF broadcast is the YOLO measurement update the UKF consumes -
                # translation only; the UKF's measurement model for this update should
                # not read rotation/yaw from it.
            else:
                self.get_logger().debug(
                    "YOLO detection present but camera isn't looking at the ground "
                    "- skipping ground-plane estimate this frame",
                    throttle_duration_sec=1.0,
                )

        # Pipeline Latency Diagnostics
        if self.diagnostics_enabled:
            transform_ready_time = self.get_clock().now()
            timing_msg = Vector3Stamped()
            timing_msg.header.stamp = stamp # t0 same as the TF's stamp, aligned
            timing_msg.header.frame_id = "yolo_pipeline_timing"
            timing_msg.vector.x = self._img_received_time.nanoseconds / 1e9 # t1
            timing_msg.vector.y = transform_ready_time.nanoseconds / 1e9    # t2
            self._yolo_pipeline_timing_publisher.publish(timing_msg)

        if self.show_debug_window:
            self._show_yolo_debug(frame, detection, translation)
            cv2.imshow("YOLO Debug", frame)
            cv2.waitKey(1)


    def _apriltag_detection(self, frame):
        """Run AprilTag detection and return recognised tag detections.

        :param frame: BGR image frame.
        :return: List of recognised AprilTag detections.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray_frame) # type: ignore

        # Filter to only recognised tag IDs
        valid_detections = [d for d in detections if d.tag_id in self._tags] # type: ignore
        return valid_detections


    def _show_apriltag_debug(self, frame, detections):
        """ Show apriltag debug.

        :param frame: Image frame
        :param detections: Apriltag detections list
        """
        for d in detections: # type: ignore
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


    def _yolo_detection(self, frame):
        """ Run YOLO and return the best landing-pad bounding box.

        :param frame: Image frame
        :return: (cx, cy, width, height, confidence) in pixels of the supplied frame,
                  or None if no suitable detection exists.
        """
        if not self.yolo_enabled or self.yolo_model is None:
            return None

        results = self.yolo_model(
            frame,
            imgsz=(384, 640),   # match the static OpenVINO export exactly
            verbose=False,
            device="cpu",
        )

        best_detection = None
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.yolo_conf_threshold:
                    continue
                class_id = int(box.cls[0])

                # class 0 = 'ugv' (single-class model)
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy())

                width = x2 - x1
                height = y2 - y1

                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5

                detection = (
                    float(cx),
                    float(cy),
                    float(width),
                    float(height),
                    confidence,
                )

                if (best_detection is None or confidence > best_detection[4]):
                    best_detection = detection

        return best_detection


    def _show_yolo_debug(self, frame, detection, translation=None):
        """ Show YOLO debug.

        :param frame:       Image frame
        :param detection:   YOLO detection tuple, or None
        :param translation: Estimated [x, y, z] from _estimate_yolo_ground_position,
                             or None if unavailable
        """
        if detection is not None:
            cx, cy, bw, bh, confidence = detection

            x1 = int(cx - bw * 0.5)
            y1 = int(cy - bh * 0.5)
            x2 = int(cx + bw * 0.5)
            y2 = int(cy + bh * 0.5)

            # Draw the YOLO bounding box and centre point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2,)
            cv2.circle(frame,(int(cx), int(cy)), 5, (255, 0, 255), -1,)
            cv2.drawMarker(frame,
                (int(cx), int(cy)),
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            cv2.putText(frame,
                f"YOLO {confidence:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

            if translation is not None:
                cv2.putText(frame,
                    f"xyz: {translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}",
                    (x1, min(frame.shape[0] - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )


    def _send_servo_command(self, servo_id, pwm_value) -> None:
        """Sends servo command via Mavlink

        :param servo_id: Servo ID to actuate
        :param pwm_value: PWM value to command to servo
        """
        req = CommandLong.Request()
        req.command = 183
        req.param1 = float(servo_id)
        req.param2 = float(pwm_value)

        self.client.call_async(req)


    def _gimbal_controller(self, image_points) -> None:
        """Determines gimbal required output to centre on tag

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
        """Publish the commanded gimbal angle to the gimbal as a PWM signal

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
    def _camera_pose_in_level_frame(quad_rotation, servo_angle) -> np.ndarray:
        """ Camera pose (position + rotation) expressed in the drone-relative,
            local-level frame - the drone's own translation is excluded, only its
            attitude and the gimbal angle are applied. This is the same frame
            convention used for both the AprilTag and YOLO landing-pad transforms, and
            is factored out here so the two pipelines can't drift apart on camera
            mount/gimbal geometry.

        :param quad_rotation: Quadcopter rotation quaternion [x, y, z, w]
        :param servo_angle:   Current gimbal servo angle (degrees)
        :return: 4x4 homogeneous transform: level-frame <- camera-frame
        """

        # Quad_local -> Quad body
        T_quad_local = tf_transformations.quaternion_matrix(quad_rotation)

        # Quad_body -> Cam
        t_quad_cam = np.array([0.02, -0.01, -0.124923])
        q_quad_cam = tf_transformations.quaternion_from_euler(
            -1.5707963 + np.deg2rad(servo_angle), 0.0, -1.5707963
        )
        T_quad_cam = tf_transformations.quaternion_matrix(q_quad_cam)
        T_quad_cam[:3, 3] = t_quad_cam

        return T_quad_local @ T_quad_cam


    def _estimate_yolo_ground_position(
        self, cx, cy, quad_rotation, servo_angle, altitude
    ) -> np.ndarray | None:
        """ Back-project a YOLO bounding-box centre into a 3D position using a
            flat-ground assumption: cast the camera ray through (cx, cy) and intersect
            it with the ground plane at local z=0, using the drone's altitude to fix
            the scale (a single 2D box alone has no depth information). This gives
            (x, y, z) translation ONLY - it cannot recover any rotation, including yaw.

            ASSUMPTION: the landing pad sits at the same height as local z=0 (i.e. the
            MAVROS local-frame origin / takeoff point). If the pad is meaningfully
            higher or lower than that, this estimate will be off by that difference.

        :param cx, cy:        YOLO bounding-box centre, in pixels of the processing
                               frame (self._image_width x self._image_height)
        :param quad_rotation: Quadcopter rotation quaternion at the image's timestamp
        :param servo_angle:   Gimbal servo angle at the image's timestamp (degrees)
        :param altitude:      Drone altitude (local-frame position.z) at the image's
                               timestamp
        :return: np.ndarray [x, y, z], drone-relative in the local-level frame (same
            convention as the AprilTag transform), or None if the camera isn't looking
            toward the ground (no valid intersection - e.g. gimbal pointed level/up)
        """

        # Camera-frame ray through the pixel (pinhole model; magnitude doesn't matter,
        # only direction, so no need to un-distort/normalise beyond unit length)
        pixel_h = np.array([cx, cy, 1.0])
        ray_cam = self._camera_matrix_inv @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        T_cam_level = self._camera_pose_in_level_frame(quad_rotation, servo_angle)
        R_cam_level = T_cam_level[:3, :3]
        camera_offset_level = T_cam_level[:3, 3]

        ray_level = R_cam_level @ ray_cam

        camera_altitude_absolute = altitude + camera_offset_level[2]

        # Ray must point toward the ground (negative z, ENU-style "up") from a camera
        # that's actually above the ground plane - otherwise there's no intersection.
        if ray_level[2] >= -1e-3 or camera_altitude_absolute <= 0:
            return None

        t = -camera_altitude_absolute / ray_level[2]
        if t <= 0:
            return None

        return camera_offset_level + t * ray_level


    @staticmethod
    def _compose_base_to_landing_pad_yolo(stamp, translation) -> TransformStamped:
        """ Pack a YOLO-derived translation-only landing-pad estimate into a
            TransformStamped, using the same frame convention as the AprilTag
            transform. Rotation is published as identity since a single monocular
            detection carries no orientation information - the UKF measurement model
            for this update should only consume translation and ignore rotation/yaw.

        :param stamp:       ROS timestamp
        :param translation: np.ndarray [x, y, z], drone-relative in the local-level
                             frame (see _estimate_yolo_ground_position)
        :return: TransformStamped: local -> landing_pad_link_yolo
        """

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "local"
        tf_msg.child_frame_id = "landing_pad_link_yolo"
        tf_msg.transform.translation.x = float(translation[0])
        tf_msg.transform.translation.y = float(translation[1])
        tf_msg.transform.translation.z = float(translation[2])
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0

        return tf_msg


    @staticmethod
    def _compose_base_to_landing_pad(
        stamp, quad_rotation, servo_angle, rvec, tvec, tag_id, tag_position
    ):
        """ Compose quad_local→quad_body→cam→tag→landing_pad into a single 
            base_link→landing_pad_link transform.

        :param stamp:         ROS timestamp
        :param quad_rotation: Quadcopter rotation quaternion
        :param servo_angle:   Current gimbal servo angle (degrees)
        :param rvec:          AprilTag rotation vector (camera→tag)
        :param tvec:          AprilTag translation vector (camera→tag)
        :param tag_id:        Detected tag ID (unused here, kept for clarity)
        :param tag_position:  (x, y, z) offset of this tag on the landing pad
        :return:              TransformStamped: base_link → landing_pad_link
        """

        T_quad_cam = VisionPerception._camera_pose_in_level_frame(quad_rotation, servo_angle)

        # Cam -> Tag
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = tvec.reshape(3)

        # Tag -> landing pad
        q_tag_pad = tf_transformations.quaternion_from_euler(0.0, 0.0, 1.5707963)
        T_tag_pad = tf_transformations.quaternion_matrix(q_tag_pad)
        T_tag_pad[:3, 3] = np.array(tag_position)

        # Compose local -> landing_pad_link (T_quad_cam here is already the combined
        # level-frame camera pose from _camera_pose_in_level_frame)
        T_local_pad = T_quad_cam @ T_cam_tag @ T_tag_pad

        t_out = T_local_pad[:3, 3]
        q_out = tf_transformations.quaternion_from_matrix(T_local_pad)

        # Pack into TransformStamped
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


    # ---- DIAGNOSTICS FUNCTION CALLBACKS ----
    def _start_video_creation(self) -> None:
        """ Create directory and start video creation from camera feed. 
        """

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_dir:
            self.frames_dir = os.path.join(self.output_dir, f"frames_{timestamp}")
        else:
            # Use workspace root or current directory
            workspace_root = _get_workspace_root()
            base_dir = workspace_root if workspace_root else os.getcwd()
            self.frames_dir = os.path.join(base_dir, f"frames_{timestamp}")

        os.makedirs(self.frames_dir, exist_ok=True)
        self.get_logger().info(
            f"Frame saving ENABLED - Directory: {self.frames_dir}"
        )
        self.get_logger().info(
            f"Video creation settings - save_frames: {self.save_frames}, create_video: {self.create_video}, fps: {self.video_fps}"
        )

        # Video output filename
        self.video_filename = os.path.join(
            os.path.dirname(self.frames_dir),
            f"yolo_detection_video_{timestamp}.mp4",
        )
        self.get_logger().info(f"Video will be saved as: {self.video_filename}")


    def _save_video_from_frame(self, frame) -> None:
        """ Save current frame from camera feed into video. 
        """

        if hasattr(self, "frames_dir"):
            frame_filename = os.path.join(
                self.frames_dir, f"frame_{self._frame_count:06d}.jpg"
            )
            success = cv2.imwrite(frame_filename, frame)
            if success:
                self._saved_frames.append(frame_filename)
                self._frame_count += 1
                if self._frame_count % 100 == 0:
                    self.get_logger().info(
                        f"Saved {self._frame_count} frames so far..."
                    )
            else:
                self.get_logger().warning(
                    f"Failed to save frame {self._frame_count}"
                )
        else:
            self.get_logger().warning(
                "Frame saving enabled but frames_dir not initialized"
            )


    def _create_video_from_frames(self) -> None:
        """ Create video from saved frames.
        """

        if not (self.save_frames or self.create_video) or not self._saved_frames:
            self.get_logger().info(
                f"Video creation skipped. save_frames={self.save_frames}, create_video={self.create_video}, frames_count={len(self._saved_frames) if hasattr(self, '_saved_frames') else 0}"
            )
            return

        try:
            duration_seconds = len(self._saved_frames) / self.video_fps
            self.get_logger().info(
                f"Creating video from {len(self._saved_frames)} frames (estimated duration: {duration_seconds:.1f}s at {self.video_fps}fps)..."
            )

            first_frame = cv2.imread(self._saved_frames[0])
            if first_frame is None:
                self.get_logger().error("Could not read first frame for video creation")
                return

            height, width, layers = first_frame.shape
            self.get_logger().info(f"Video dimensions: {width}x{height}")

            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, self.video_fps, (width, height)
            )

            if not video_writer.isOpened():
                self.get_logger().error("Failed to open video writer")
                return

            frames_written = 0
            for i, frame_path in enumerate(self._saved_frames):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
                    frames_written += 1
                    if (i + 1) % 100 == 0:
                        self.get_logger().info(
                            f"Writing frame {i + 1}/{len(self._saved_frames)} to video..."
                        )
                else:
                    self.get_logger().warning(f"Could not read frame: {frame_path}")

            video_writer.release()
            self.get_logger().info(f"Video created successfully: {self.video_filename}")
            self.get_logger().info(
                f"Final video stats: {frames_written} frames written, duration: {frames_written/self.video_fps:.1f}s"
            )

            if not self.save_frames:
                self.get_logger().info("Cleaning up temporary frame files...")
                for frame_path in self._saved_frames:
                    try:
                        os.remove(frame_path)
                    except OSError as e:
                        self.get_logger().warning(
                            f"Could not remove frame {frame_path}: {e}"
                        )
                try:
                    os.rmdir(self.frames_dir)
                except OSError:
                    pass

        except Exception as e:
            self.get_logger().error(f"Error creating video: {e}")


# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = VisionPerception()

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
        if hasattr(node, "create_video_from_frames") and not shutdown_in_progress:
            node._create_video_from_frames()
        elif hasattr(node, "create_video_from_frames"):
            try:
                node.get_logger().info("Creating video during shutdown...")
                node._create_video_from_frames()
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