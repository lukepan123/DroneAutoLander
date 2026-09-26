import os
import sys
import signal
import cv2
import numpy as np
import tf2_ros
import rclpy
import logging
import warnings

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

from ultralytics import YOLO
from ultralytics import utils as yutils

from .vision_common import CameraIntrinsics
from .vision_common import OdometryBuffer
from .vision_common import GimbalAngleBuffer
from .vision_common import FrameRecorder
from .vision_common import get_workspace_root
from .vision_common import camera_pose_in_level_frame
from .vision_common import pixel_row_to_angle_error
from .vision_common import stamp_to_sec

""" YOLO Landing Pad Detection Node.

    Runs the coarser, heavier perception loop: monocular 'ugv' bounding-box detection
    -> flat-ground back-projection -> base_link -> landing_pad_link_yolo TF broadcast.
    This TF broadcast is a second, independent measurement stream feeding the same UKF
    that apriltag.py's TF also feeds (translation only - see
    _estimate_yolo_ground_position for why this can't recover rotation/yaw).

    Split out from the combined vision_perception node into its own process so a slow
    YOLO inference tick (OpenVINO on CPU) can never stall apriltag.py's gimbal
    control / TF broadcast loop - previously both pipelines ran on one executor
    thread and contended for it.

    This node does not command the gimbal servo itself - apriltag.py owns the only
    actuator. It reads apriltag.py's published servo angle off
    /landing_pad/gimbal_angle and time-interpolates it (see GimbalAngleBuffer) to
    recover the angle that was actually in effect when its own frame was captured,
    for use in _estimate_yolo_ground_position. It also publishes its own detection's
    vertical centring error on /landing_pad/yolo_gimbal_error so apriltag.py can use
    it as a FALLBACK gimbal-control input on ticks where its own AprilTag detection
    misses - AprilTag detection always takes priority when available (see
    apriltag.py's _apriltag_timer_callback / _drive_gimbal).
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


class YoloNode(Node):
    """ Defines the YOLO perception node. """

    def __init__(self) -> None:
        """ Initialise the YOLO perception node. """

        super().__init__("yolo_landing_pad_node")

        # ---- NODE PARAMETERS ----
        self.declare_parameter("diagnostics_enabled", True)
        self.diagnostics_enabled = (
            self.get_parameter("diagnostics_enabled").get_parameter_value().bool_value
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

        self.declare_parameter("video_fps", 10.0)
        self.video_fps = float(
            self.get_parameter("video_fps").get_parameter_value().double_value
        )

        self.declare_parameter("output_dir", "")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        self.declare_parameter("yolo_processing_rate", 10.0)
        self._yolo_processing_rate = float(
            self.get_parameter("yolo_processing_rate").get_parameter_value().double_value
        )

        self.declare_parameter("imgsz_width", 640)
        self._image_width = int(
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )

        self.declare_parameter("imgsz_height", 384)
        self._image_height = int(
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )

        self.declare_parameter("landing_pad_class_id", 1)
        self._landing_pad_class_id = int(
            self.get_parameter("landing_pad_class_id").get_parameter_value().integer_value
        )

        self.declare_parameter("car_class_id", 0)
        self._car_class_id = int(
            self.get_parameter("car_class_id").get_parameter_value().integer_value
        )

        self.get_logger().info(f"YOLO processing rate: {self._yolo_processing_rate} Hz")
        self.get_logger().info(
            f"Video recording parameters: save_frames={self.save_frames}, "
            f"create_video={self.create_video}, video_fps={self.video_fps}"
        )

        self.declare_parameter("ground_z", 1.5)
        self._ground_z = float(
            self.get_parameter("ground_z").get_parameter_value().double_value
        )
        self.get_logger().info(f"Ground plane Z: {self._ground_z:.3f} m")

        # ---- CAMERA INTRINSICS ----
        # Needed to back-project YOLO pixel detections into camera-frame rays (no
        # known object size for YOLO, so we can't solvePnP like AprilTag does - see
        # _estimate_yolo_ground_position). Also used for the vertical
        # pixel-row-to-angle-error gimbal fallback published below.
        self._intrinsics = CameraIntrinsics(self._image_width, self._image_height)
        self._camera_matrix_inv = self._intrinsics.matrix_inv

        # ---- YOLO MODEL ----
        workspace_root = get_workspace_root()
        default_yolo_model = (
            os.path.join(workspace_root, "ugv_sim_yolo11n_openvino_model")
            if workspace_root
            else "ugv_sim_yolo11n_openvino_model"
        )

        self.declare_parameter("yolo_enabled", True)
        self.declare_parameter("yolo_model_path", default_yolo_model)
        self.declare_parameter("yolo_conf_threshold", 0.60)

        self.yolo_enabled = (
            self.get_parameter("yolo_enabled").get_parameter_value().bool_value
        )
        self.yolo_model_path = (
            self.get_parameter("yolo_model_path").get_parameter_value().string_value
        )
        self.yolo_conf_threshold = float(
            self.get_parameter("yolo_conf_threshold").get_parameter_value().double_value
        )

        self.yolo_model = None
        if self.yolo_enabled:
            try:
                self.get_logger().info(f"Loading YOLO model: {self.yolo_model_path}")
                self.yolo_model = YOLO(self.yolo_model_path)
                self.get_logger().info(
                    f"YOLO model loaded successfully. Classes: {self.yolo_model.names}"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to load YOLO model '{self.yolo_model_path}': {e}"
                )
                raise

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

        # apriltag.py owns the gimbal; this node reads its published angle back and
        # time-aligns it to its own frames.
        self._gimbal_angle_sub = self.create_subscription(
            Vector3Stamped, "/landing_pad/gimbal_angle", self._gimbal_angle_callback, 10
        )
        self._gimbal_buffer = GimbalAngleBuffer(window_s=1.0)
        # Used only if no /landing_pad/gimbal_angle messages have arrived yet (e.g.
        # apriltag.py not running / just started) - matches the original node's
        # startup default.
        self._fallback_servo_angle = -90.0

        _img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_subscription = self.create_subscription(
            Image, "/camera/image_raw", self._image_store_callback, _img_qos
        )
        self.get_logger().info(
            "YOLO node started, waiting for /camera/image_raw (published directly, "
            "or republished by apriltag.py when it's running in webcam mode)..."
        )

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._landing_pad_found_publisher = self.create_publisher(
            Bool, "/landing_pad/found", 10
        )
        # Fallback gimbal-control input for apriltag.py - see module docstring above
        # and apriltag.py's _apriltag_timer_callback / _drive_gimbal.
        self._yolo_gimbal_error_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/yolo_gimbal_error", 10
        )

        # ---- TF2 ----
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- DIAGNOSTICS ----
        self._yolo_pipeline_timing_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/yolo_pipeline_timing", 10
        )

        # ---- FRAME RECORDING ----
        self._frame_recorder = FrameRecorder(
            self.get_logger(), "yolo", self.output_dir, self.video_fps,
            self.save_frames, self.create_video,
        )

        if self.show_debug_window:
            cv2.namedWindow("YOLO Debug", cv2.WINDOW_AUTOSIZE)

        # ---- IMAGE BUFFER ----
        self._img_msg = None
        self._img_received_time = None
        self._img_seq = 0
        self._last_processed_seq = -1

        # ---- PROCESSING TIMER ----
        self._yolo_timer = self.create_timer(
            1.0 / self._yolo_processing_rate, self._yolo_timer_callback
        )

    # ---- RECEPTION CALLBACKS ----
    def _image_store_callback(self, msg) -> None:
        """ Store the latest incoming image message.

        :param msg: Incoming Image message from /camera/image_raw
        """
        self._img_msg = msg
        self._img_received_time = self.get_clock().now()
        self._img_seq += 1

    def _quad_odometry_callback(self, msg: Odometry) -> None:
        """ Obtain FCU Odometry, buffering stamped attitude/altitude for
            time-correct lookups.

        :param msg: Incoming Odometry message
        """
        self._odom_buffer.push(msg)

    def _gimbal_angle_callback(self, msg: Vector3Stamped) -> None:
        """ Buffer apriltag.py's published gimbal angle for later time-alignment
            against this node's own frame timestamps.

        :param msg: Vector3Stamped with the servo angle (degrees) in vector.x
        """
        t = stamp_to_sec(msg.header.stamp)
        self._gimbal_buffer.push(t, float(msg.vector.x))

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
                interpolation=cv2.INTER_AREA,
            )

        return frame, stamp

    def _yolo_timer_callback(self) -> None:
        """ Fires at yolo_processing_rate. Runs the coarse YOLO landing-pad detector
            independently of the AprilTag pipeline and publishes its own measurement
            update to the UKF (via TF), plus a fallback gimbal-centring error for
            apriltag.py to use on ticks where its own detection misses.
        """

        if not self.yolo_enabled or self.yolo_model is None:
            return

        frame, stamp = self._get_new_frame()
        if frame is None:
            return  # no new frame since this node last ran

        odom = self._odom_buffer.get_at(stamp)
        if odom is None:
            self.get_logger().warn(
                "No odometry buffered yet - skipping frame", throttle_duration_sec=1.0
            )
            return
        q, altitude = odom

        servo_angle = self._gimbal_buffer.get_at(stamp)
        if servo_angle is None:
            servo_angle = self._fallback_servo_angle
            self.get_logger().warn(
                "No gimbal angle received from apriltag.py yet - using fallback",
                throttle_duration_sec=2.0,
            )

        detections = self._yolo_detections(frame)

        landing_pad_detection = detections["landing_pad"]
        landing_pad_translation = None
        car_detection = detections["car"]
        car_translation = None

        # Push a gimbal correction, prioritising centring on the landing_pad vs the car
        any_detection = None
        if landing_pad_detection is not None:
            any_detection = landing_pad_detection
        elif car_detection is not None:
            any_detection = car_detection

        if any_detection is not None:
            cx, cy, bw, bh, confidence = any_detection #type: ignore
            # Publish this detection's vertical centring error immediately, ahead of
            # the ground-plane back-projection below - the gimbal fallback only needs
            # the bounding-box centre and is still valid even on frames where the
            # ground-plane intersection fails (e.g. gimbal pointed too level).
            fy = self._intrinsics.matrix[1, 1]
            angle_error = pixel_row_to_angle_error(cy, self._image_height, fy)
            gimbal_error_msg = Vector3Stamped()
            gimbal_error_msg.header.stamp = stamp
            gimbal_error_msg.header.frame_id = "yolo_gimbal_error"
            gimbal_error_msg.vector.x = angle_error
            gimbal_error_msg.vector.y = confidence
            self._yolo_gimbal_error_publisher.publish(gimbal_error_msg)

        if landing_pad_detection is not None:
            cx, cy, bw, bh, confidence = landing_pad_detection #type: ignore

            # A single 2D box gives a bearing to the pad, not depth or orientation,
            # so we recover (x, y, z) via a flat-ground assumption: cast the ray
            # through the box centre and intersect it with the ground plane, using
            # altitude to fix the scale. This carries no rotation information - see
            # _compose_base_to_landing_pad_yolo, which publishes identity rotation.
            landing_pad_translation = self._estimate_yolo_ground_position(
                cx, cy, q, servo_angle, altitude
            )

            if landing_pad_translation is not None:
                # If we have a detection, then the landing pad is found. Note: unlike
                # apriltag.py, this branch only ever publishes True - it never asserts
                # "not found", since a missed YOLO detection on any given tick shouldn't
                # override AprilTag's own found/not-found signal on the shared topic.
                self._landing_pad_found_publisher.publish(Bool(data=True))
                tf_base_to_pad_yolo = self._compose_base_to_landing_pad_yolo(
                    stamp, landing_pad_translation
                )
                self._tf_broadcaster.sendTransform(tf_base_to_pad_yolo)
                # This TF broadcast is the YOLO measurement update the UKF consumes -
                # translation only; the UKF's measurement model for this update
                # should not read rotation/yaw from it.
            else:
                self.get_logger().debug(
                    "YOLO detection present but camera isn't looking at the ground "
                    "- skipping ground-plane estimate this frame",
                    throttle_duration_sec=1.0,
                )

        if car_detection is not None:
            cx, cy, bw, bh, confidence = car_detection #type: ignore

            car_translation = self._estimate_yolo_ground_position(
                cx, cy, q, servo_angle, altitude,
            )

            if car_translation is not None:
                self._landing_pad_found_publisher.publish(Bool(data=True))
                tf_base_to_car_yolo = self._compose_base_to_car_yolo(
                    stamp, car_translation,
                )
                self._tf_broadcaster.sendTransform(tf_base_to_car_yolo)

        # Pipeline Latency Diagnostics
        if self.diagnostics_enabled:
            transform_ready_time = self.get_clock().now()
            timing_msg = Vector3Stamped()
            timing_msg.header.stamp = stamp
            timing_msg.header.frame_id = "yolo_pipeline_timing"
            timing_msg.vector.x = self._img_received_time.nanoseconds / 1e9 #type: ignore
            timing_msg.vector.y = transform_ready_time.nanoseconds / 1e9
            self._yolo_pipeline_timing_publisher.publish(timing_msg)

        if self.show_debug_window:
            self._show_yolo_debug(
                frame,
                landing_pad_detection,
                car_detection,
                landing_pad_translation,
                car_translation,
            )
            cv2.imshow("YOLO Debug", frame)
            cv2.waitKey(1)

        if self.save_frames or self.create_video:
            self._frame_recorder.save(frame)

    def _yolo_detections(self, frame):
        """Run YOLO and return the best detection for each supported class.
        Returns:
            {
                "landing_pad": (cx, cy, width, height, confidence),
                "car":         (cx, cy, width, height, confidence),
            }
            Missing classes have value None.
        """

        detections = {"landing_pad": None, "car": None}

        if not self.yolo_enabled or self.yolo_model is None:
            return detections

        results = self.yolo_model(frame, imgsz=(384, 640), verbose=False, device="cpu")

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence < self.yolo_conf_threshold:
                    continue

                class_id = int(box.cls[0])

                if class_id == self._landing_pad_class_id:
                    key = "landing_pad"
                elif class_id == self._car_class_id:
                    key = "car"
                else:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width = x2 - x1
                height = y2 - y1
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                detection = (float(cx), float(cy), float(width), float(height), confidence)

                # Keep highest-confidence detection for each class
                previous = detections[key]

                if previous is None or confidence > previous[4]:
                    detections[key] = detection #type: ignore

        return detections

    def _show_yolo_debug(
        self,
        frame,
        landing_pad_detection,
        car_detection,
        landing_pad_translation=None,
        car_translation=None,
    ):
        """Show YOLO debug detections."""

        def draw_detection(detection, translation, label, box_color):
            """ In-line helper function to debug YOLO detections"""
            if detection is None:
                return

            cx, cy, bw, bh, confidence = detection

            x1 = int(cx - bw * 0.5)
            y1 = int(cy - bh * 0.5)
            x2 = int(cx + bw * 0.5)
            y2 = int(cy + bh * 0.5)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Centre point
            cv2.circle(frame, (int(cx), int(cy)), 5, box_color, -1)

            # Crosshair
            cv2.drawMarker(frame, (int(cx), int(cy)), box_color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # Label
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

            # Estimated XYZ
            if translation is not None:
                cv2.putText(
                    frame,
                    (
                        f"xyz: "
                        f"{translation[0]:.2f}, "
                        f"{translation[1]:.2f}, "
                        f"{translation[2]:.2f}"
                    ),
                    (x1, min(frame.shape[0] - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                )

        # Draw landing pad
        draw_detection(
            landing_pad_detection,
            landing_pad_translation,
            "Landing pad",
            (255, 0, 255),
        )

        # Draw car
        draw_detection(
            car_detection,
            car_translation,
            "Car",
            (0, 255, 255),
        )

    def _estimate_yolo_ground_position(
        self, cx, cy, quad_rotation, servo_angle, altitude
    ) -> np.ndarray | None:
        """ Back-project a YOLO bounding-box centre into a 3D position using a
            flat-ground assumption: cast the camera ray through (cx, cy) and
            intersect it with the ground plane at local z=0, using the drone's
            altitude to fix the scale (a single 2D box alone has no depth
            information). This gives (x, y, z) translation ONLY - it cannot recover
            any rotation, including yaw.

            ASSUMPTION: the landing pad sits at the same height as local z=0 (i.e.
            the MAVROS local-frame origin / takeoff point). If the pad is
            meaningfully higher or lower than that, this estimate will be off by
            that difference.

        :param cx, cy:        YOLO bounding-box centre, in pixels of the processing
                               frame (self._image_width x self._image_height)
        :param quad_rotation: Quadcopter rotation quaternion at the image's timestamp
        :param servo_angle:   Gimbal servo angle at the image's timestamp (degrees),
                               as reported by apriltag.py on /landing_pad/gimbal_angle
        :param altitude:      Drone altitude (local-frame position.z) at the image's
                               timestamp
        :return: np.ndarray [x, y, z], drone-relative in the local-level frame (same
            convention as the AprilTag transform), or None if the camera isn't
            looking toward the ground (no valid intersection - e.g. gimbal pointed
            level/up)
        """

        pixel_h = np.array([cx, cy, 1.0])
        ray_cam = self._camera_matrix_inv @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        T_cam_level = camera_pose_in_level_frame(quad_rotation, servo_angle)
        R_cam_level = T_cam_level[:3, :3]
        camera_offset_level = T_cam_level[:3, 3]

        ray_level = R_cam_level @ ray_cam

        camera_altitude_absolute = altitude + camera_offset_level[2]

        camera_z = altitude + camera_offset_level[2]

        if ray_level[2] >= -1e-3:
            return None

        t = (self._ground_z - camera_z) / ray_level[2]
        if t <= 0:
            return None

        return camera_offset_level + t * ray_level

    @staticmethod
    def _compose_base_to_landing_pad_yolo(stamp, translation) -> TransformStamped:
        """ Pack a YOLO-derived translation-only landing-pad estimate into a
            TransformStamped, using the same frame convention as the AprilTag
            transform. Rotation is published as identity since a single monocular
            detection carries no orientation information.

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
    def _compose_base_to_car_yolo(stamp, translation) -> TransformStamped:
        """ Pack a YOLO car detection into a translation-only TF.
        """

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "local"
        tf_msg.child_frame_id = "car_link_yolo"
        tf_msg.transform.translation.x = float(translation[0])
        tf_msg.transform.translation.y = float(translation[1])
        tf_msg.transform.translation.z = float(translation[2])
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0

        return tf_msg

# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

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

        node.destroy_node()
        if getattr(node, "show_debug_window", True):
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()