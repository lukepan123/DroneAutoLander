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
from scipy.optimize import least_squares
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
from .vision_common import camera_pose_in_level_frame
from .vision_common import image_target_message
from .vision_common import load_camera_mount
from .vision_common import stamp_to_sec
from .vision_common import get_workspace_root

""" YOLO Landing Pad Detection Node.

    Runs the coarser, heavier perception loop: monocular 'ugv' bounding-box detection
    -> flat-ground back-projection -> base_link -> landing_pad_link_yolo TF broadcast.
    This TF broadcast is a second, independent measurement stream feeding the same UKF
    that apriltag.py's TF also feeds (translation only - see
    _estimate_yolo_ground_position for why this can't recover rotation/yaw).

    This node does not command the gimbal servo. It publishes the best available image
    target on /landing_pad/yolo_target for gimbal_controller.py, preferring the landing
    pad class and using the car class as a coarse fallback. The gimbal controller owns
    the actuator independently, so YOLO can continue to drive the gimbal even when the
    AprilTag node is disabled.

    Split out from the combined vision_perception node into its own process so a slow
    YOLO inference tick (OpenVINO on CPU) can never stall apriltag.py's detector or the
    independent gimbal controller.

    For the ground-plane estimate, this node builds the camera pose itself at the image
    timestamp: buffered FCU odometry (attitude + altitude) and the gimbal angle from
    /landing_pad/gimbal_angle are both interpolated to the image stamp and fed through
    camera_pose_in_level_frame. There is no camera TF lookup, so there is nothing to
    extrapolate and no frame is dropped because a transform arrived a few ms late.
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
        self.diagnostics_enabled = (self.get_parameter("diagnostics_enabled").get_parameter_value().bool_value)
        self.declare_parameter("show_debug_window", False)
        self.show_debug_window = (self.get_parameter("show_debug_window").get_parameter_value().bool_value)

        # ---- RECORDING PARAMETERS ----
        self.declare_parameter("save_frames", False)
        self.save_frames = (self.get_parameter("save_frames").get_parameter_value().bool_value)
        self.declare_parameter("create_video", True)
        self.create_video = (self.get_parameter("create_video").get_parameter_value().bool_value)
        self.declare_parameter("video_fps", 10.0)
        self.video_fps = (self.get_parameter("video_fps").get_parameter_value().double_value)
        self.declare_parameter("output_dir", "")
        self.output_dir = (self.get_parameter("output_dir").get_parameter_value().string_value)

        # ---- PROCESSING PARAMETERS ----
        self.declare_parameter("yolo_processing_rate", 10.0)
        self._yolo_processing_rate = (self.get_parameter("yolo_processing_rate").get_parameter_value().double_value)
        self.declare_parameter("img_width", 640)
        self._image_width = (self.get_parameter("img_width").get_parameter_value().integer_value)
        self.declare_parameter("img_height", 360)
        self._image_height = (self.get_parameter("img_height").get_parameter_value().integer_value)
        self.declare_parameter("yolo_img_width", 640)
        self._yolo_image_width = (self.get_parameter("yolo_img_width").get_parameter_value().integer_value)
        self.declare_parameter("yolo_img_height", 384)
        self._yolo_image_height = (self.get_parameter("yolo_img_height").get_parameter_value().integer_value)

        # ---- YOLO PARAMETERS ----
        workspace_root = get_workspace_root()
        default_yolo_model = (os.path.join(workspace_root, "ugv_yolo11n_openvino_model") if workspace_root else "ugv_yolo11n_openvino_model")
        self.declare_parameter("yolo_enabled", True)
        self.yolo_enabled = (self.get_parameter("yolo_enabled").get_parameter_value().bool_value)
        self.declare_parameter("yolo_model_path", default_yolo_model)
        self.yolo_model_path = (self.get_parameter("yolo_model_path").get_parameter_value().string_value)

        self.declare_parameter("yolo_conf_threshold", 0.60)
        self.yolo_conf_threshold = (self.get_parameter("yolo_conf_threshold").get_parameter_value().double_value)
        self.declare_parameter("yolo_landing_pad_class_id", 1)
        self._landing_pad_class_id = (self.get_parameter("yolo_landing_pad_class_id").get_parameter_value().integer_value)
        self.declare_parameter("yolo_car_class_id", 0)
        self._car_class_id = (self.get_parameter("yolo_car_class_id").get_parameter_value().integer_value)

        self.declare_parameter("yolo_edge_margin_px", 0)
        self._yolo_edge_margin_px = (self.get_parameter("yolo_edge_margin_px").get_parameter_value().integer_value)

        # ---- YOLO OBJECT PARAMETERS ----
        self.declare_parameter("yolo_car_roof_AGL", 1.5)
        self._landing_pad_z = (self.get_parameter("yolo_car_roof_AGL").get_parameter_value().double_value)
        self.declare_parameter("yolo_ground_ray_z_epsilon", 1e-3)
        self._ground_ray_z_epsilon = (self.get_parameter("yolo_ground_ray_z_epsilon").get_parameter_value().double_value)
        self.declare_parameter("yolo_car_size_3d", [1.8, 4.3, 0.85])  # car extent (m): width, length, height (of body NOT car)
        self._car_size_3d = np.array(self.get_parameter("yolo_car_size_3d").get_parameter_value().double_array_value)
        self.declare_parameter("yolo_car_fit_max_rel_rms", 0.08)  # max 3D-box fit residual, as a fraction of box size
        self._car_fit_max_rel_rms = self.get_parameter("yolo_car_fit_max_rel_rms").get_parameter_value().double_value
        self.declare_parameter("yolo_roof_height_gain", 0.2)  # per car detection; how fast the roof height tracks the fit (0 = fixed)
        self._roof_height_gain = self.get_parameter("yolo_roof_height_gain").get_parameter_value().double_value
        self.declare_parameter("yolo_car_roof_above_body", 0.65)  # roof height minus body-top height (m)
        self._car_roof_above_body = self.get_parameter("yolo_car_roof_above_body").get_parameter_value().double_value


        # ---- CAMERA PARAMETERS ----
        self.declare_parameter("camera_fov_horizontal", 2.7925268)
        self._camera_fov_horizontal = (self.get_parameter("camera_fov_horizontal").get_parameter_value().double_value)
        self.declare_parameter("camera_distortion_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        camera_distortion_coeffs = (self.get_parameter("camera_distortion_coeffs").get_parameter_value().double_array_value)
        if len(camera_distortion_coeffs) != 5:
            raise ValueError("camera_distortion_coeffs must contain exactly five values")

        # ---- BUFFER PARAMETERS ----
        self.declare_parameter("odometry_buffer_window_s", 1.0)
        self._odometry_buffer_window_s = (self.get_parameter("odometry_buffer_window_s").get_parameter_value().double_value)


        self.get_logger().info(f"YOLO processing rate: {self._yolo_processing_rate} Hz")
        self.get_logger().info(
            f"YOLO parameters: model={self.yolo_model_path}, conf={self.yolo_conf_threshold:.2f}, "
            f"input={self._image_width}x{self._image_height}, "
            f"landing_pad_id={self._landing_pad_class_id}, car_id={self._car_class_id}"
        )
        self.get_logger().info(f"Roof plane Z above home altitude: {self._landing_pad_z:.3f} m")
        self.get_logger().info(
            f"Video recording parameters: save_frames={self.save_frames}, "
            f"create_video={self.create_video}, video_fps={self.video_fps}"
        )

        # ---- CAMERA INTRINSICS ----
        self._intrinsics = CameraIntrinsics(
            self._image_width,
            self._image_height,
            self._camera_fov_horizontal,
            tuple(camera_distortion_coeffs), # type:ignore
        )
        self._camera_matrix_inv = self._intrinsics.matrix_inv
        self._camera_matrix = np.linalg.inv(self._camera_matrix_inv)

        # ---- YOLO MODEL ----
        self.yolo_model = None
        if self.yolo_enabled:
            try:
                self.get_logger().info(f"Loading YOLO model: {self.yolo_model_path}")
                self.yolo_model = YOLO(self.yolo_model_path)
                self.get_logger().info(f"YOLO model loaded successfully. Classes: {self.yolo_model.names}")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model '{self.yolo_model_path}': {e}")
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

        # Camera mount (shared defaults live in vision_common.load_camera_mount)
        self._camera_mount = load_camera_mount(self)

        # ---- TF2 ----
        # Broadcast only: this node publishes its landing pad / car measurements as TF
        # but no longer listens to /tf.

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
        self._landing_pad_found_publisher = self.create_publisher(Bool, "/landing_pad/found", 10)
        self._yolo_target_publisher = self.create_publisher(Vector3Stamped, "/landing_pad/yolo_target", 10)

        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- DIAGNOSTICS ----
        self._yolo_pipeline_timing_publisher = self.create_publisher(
            Vector3Stamped, "/landing_pad/yolo_pipeline_timing", 10
        )

        # ---- FRAME RECORDING ----
        self._frame_recorder = FrameRecorder(
            self.get_logger(),
            "yolo",
            self.output_dir,
            self.video_fps,
            self.save_frames,
            self.create_video,
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
        """ Buffer the measured gimbal pitch for image-timestamp alignment.

        :param msg: Vector3Stamped from gimbal_controller.py; vector.x is the actual
                    gimbal pitch in degrees
        """
        self._gimbal_buffer.push(stamp_to_sec(msg.header.stamp), msg.vector.x)

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
            update to the UKF (via TF), plus an image target for the independent gimbal
            controller. The camera pose used for ground back-projection is built locally
            from time-aligned odometry and gimbal angle.
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

        gimbal_deg = self._gimbal_buffer.get_at(stamp)
        if gimbal_deg is None:
            self.get_logger().warn(
                "No gimbal angle buffered yet - skipping frame",
                throttle_duration_sec=1.0,
            )
            return

        # Camera pose in the local-level frame at the image timestamp (attitude and
        # gimbal angle both interpolated to `stamp`).
        T_local_cam = camera_pose_in_level_frame(q, gimbal_deg, self._camera_mount)

        detections = self._yolo_detections(frame)

        landing_pad_detection = detections["landing_pad"]
        landing_pad_translation = None
        car_detection = detections["car"]
        car_translation = None

        # Push a gimbal target, prioritising the landing pad over the coarse car class.
        any_detection = None
        if landing_pad_detection is not None:
            any_detection = landing_pad_detection
        elif car_detection is not None:
            any_detection = car_detection

        if any_detection is not None:
            cx, cy, _bw, _bh, _confidence = any_detection  #type: ignore
            self._yolo_target_publisher.publish(
                image_target_message(
                    stamp, cx, cy, self._intrinsics, "yolo_target"
                )
            )

        if landing_pad_detection is not None:
            cx, cy, bw, bh, confidence = landing_pad_detection #type: ignore

            # A single 2D box gives a bearing to the pad, not depth or orientation,
            # so we recover (x, y, z) via a flat-ground assumption: cast the ray
            # through the box centre and intersect it with the ground plane, using
            # altitude to fix the scale. This carries no rotation information - see
            # _compose_base_to_landing_pad_yolo, which publishes identity rotation.
            landing_pad_translation = self._estimate_yolo_ground_position(
                cx, cy, T_local_cam, altitude
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

        if car_detection is not None:
            cx, cy, bw, bh, confidence = car_detection #type: ignore

            car_translation = self._estimate_yolo_ground_position(
                cx, cy, T_local_cam, altitude, kind="car", bw=bw, bh=bh
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
            timing_msg.vector.x = self._img_received_time.nanoseconds / 1e9  #type: ignore
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
                "car": (cx, cy, width, height, confidence),
            }
            Missing classes have value None.
        """

        detections = {"landing_pad": None, "car": None}

        if not self.yolo_enabled or self.yolo_model is None:
            return detections

        results = self.yolo_model(
            frame,                                   # 640x360
            imgsz=(self._yolo_image_height, self._yolo_image_width), # (384, 640)
            verbose=False,
            device="cpu",
        )

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
                margin = self._yolo_edge_margin_px

                if (
                    x1 < margin
                    or y1 < margin
                    or x2 > self._image_width - margin
                    or y2 > self._image_height - margin
                ):
                    continue

                width = x2 - x1
                height = y2 - y1
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                detection = (float(cx), float(cy), float(width), float(height), confidence)

                # Keep highest-confidence detection for each class
                previous = detections[key]

                if previous is None or confidence > previous[4]:
                    detections[key] = detection  #type: ignore

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
            cv2.drawMarker(
                frame,
                (int(cx), int(cy)),
                box_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

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
        self, cx, cy, T_local_cam, altitude, kind="landing_pad", bw=None, bh=None
    ) -> np.ndarray | None:
        """ Back-project a YOLO bounding box into a drone-relative 3D position
            (translation only).

            kind="landing_pad": intersect the box-centre ray with the roof plane.
            kind="car": intersect the box-centre ray with the car's mid-height plane, then,
                if bw/bh are given, refine by fitting a car-sized 3D box so that its
                projected bounding box matches the YOLO box. This stays valid for oblique
                views (foreshortening, visible roof + sides), unlike a size-based depth.
                The car's yaw relative to the camera heading is fitted too, with a weak
                prior towards "aligned". The box size also reveals if the roof is higher or
                lower than self._landing_pad_z (car on a slope, altitude drift); that offset
                is low-passed back into self._landing_pad_z, which the pad estimate uses too.
                Only car detections correct the roof height; the pad has no known size.

        :return: np.ndarray [x, y, z] in the local-level frame, or None if the camera isn't
                looking toward the ground
        """

        R = T_local_cam[:3, :3]
        cam = T_local_cam[:3, 3]  # camera position relative to the drone

        ray = R @ self._camera_matrix_inv @ np.array([cx, cy, 1.0])
        ray /= np.linalg.norm(ray)
        if ray[2] >= -self._ground_ray_z_epsilon:
            return None

        # Plane to intersect: the roof for the pad, the car's mid-height for the car
        plane_z = self._landing_pad_z - ((self._car_roof_above_body + 0.5 * self._car_size_3d[2]) if kind == "car" else 0.0)
        t = (plane_z - (altitude + cam[2])) / ray[2]
        if t <= 0:
            return None
        pos = cam + t * ray
        if kind != "car":
            return pos

        roof_offset = 0.5 * self._car_size_3d[2] + self._car_roof_above_body
        if bw is None or bh is None:
            return pos + np.array([0.0, 0.0, roof_offset])

        # Car footprint axes, assuming the car is yaw-aligned with the camera heading:
        # width along the image x-axis, length along the perpendicular horizontal axis.
        u_w = np.array([R[0, 0], R[1, 0], 0.0])
        if np.linalg.norm(u_w) < 0.2:  # camera x-axis ~vertical: heading undefined
            return pos + np.array([0.0, 0.0, roof_offset])
        u_w /= np.linalg.norm(u_w)
        u_l = np.array([-u_w[1], u_w[0], 0.0])

        # 8 box corners in the car frame (width, length, height) and the YOLO box edges
        corners = np.array([[a, b, c] for a in (-.5, .5) for b in (-.5, .5) for c in (-.5, .5)])
        corners = corners * self._car_size_3d
        box = np.array([cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2])

        k_dz = 3.0  # height prior (px per metre): weak in pass 1, ~pinned in pass 2

        def residual(p):  # p = [x, y, yaw offset, height offset] -> projected-box edges minus YOLO box edges
            c, s = np.cos(p[2]), np.sin(p[2])
            axes = np.array([c * u_w + s * u_l, c * u_l - s * u_w, [0.0, 0.0, 1.0]])
            pts = (np.array([p[0], p[1], pos[2] + p[3]]) + corners @ axes - cam) @ R  # camera frame
            uv = pts @ self._camera_matrix.T
            uv = uv[:, :2] / np.maximum(uv[:, 2:], 0.1)
            return np.array([
                uv[:, 0].min() - box[0], uv[:, 0].max() - box[1],
                uv[:, 1].min() - box[2], uv[:, 1].max() - box[3],
                np.degrees(p[2]) / 5.0,  # weak yaw prior: costs 1 px per 5 deg
                k_dz * p[3],
            ])

        def fit():
            # Fit from a few yaw guesses (one start gets stuck at "aligned"). The sign of the
            # yaw is not observable from a box, so blend the fits by likelihood (~2 px edge
            # noise) instead of picking one at random: ambiguous -> midpoint, clear -> best.
            fits = [least_squares(residual, [pos[0], pos[1], yaw0, 0.0], method="lm", max_nfev=25, ftol=1e-3, xtol=1e-3)
                    for yaw0 in (-0.3, 0.3)]
            cost = np.array([f.cost for f in fits])
            w = np.exp(-(cost - cost.min()) / 4.0)
            rms = np.sqrt(np.mean(fits[cost.argmin()].fun[:4] ** 2))
            return w @ np.array([f.x for f in fits]) / w.sum(), rms

        # Pass 1: let the height float to learn where the roof really is (slope, altitude
        # drift) and low-pass that into self._landing_pad_z.
        (x, y, _, dz), rms = fit()
        if rms > max(self._car_fit_max_rel_rms * 0.5 * (bw + bh), 3.0):
            return pos + np.array([0.0, 0.0, roof_offset])  # box doesn't fit the car model (bad yaw / bad detection)
        self._landing_pad_z += self._roof_height_gain * dz
        pos = np.array([x, y, pos[2] + self._roof_height_gain * dz])

        self.get_logger().info(
            f"car fit: rms={rms:.1f}px (limit {self._car_fit_max_rel_rms * 0.5 * (bw + bh):.1f}) "
            f"bw={bw:.0f} bh={bh:.0f} roof={self._landing_pad_z:.2f} alt={altitude:.2f}",
            throttle_duration_sec=1.0)

        # Pass 2: fit x, y with that height ~fixed (a free height adds noise to x, y in
        # oblique views, because height and range are coupled).
        k_dz = 100.0
        (x, y, _, _), _ = fit()
        return np.array([x, y, self._landing_pad_z - altitude])


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
        """ Pack a YOLO car detection into a translation-only TF. """

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