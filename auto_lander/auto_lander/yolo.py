import os
import sys
import signal
import threading
import cv2
import numpy as np
import rclpy
import logging
import warnings

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped

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

""" YOLO Car Detection Node.

    Runs the coarser, heavier perception loop: monocular 'car' bounding-box detection.
    This node no longer back-projects the box itself. It publishes the RAW box together
    with the camera geometry at the image timestamp, and the UKF (controller.py) does
    the 3D-box measurement update, where it can use the live landing-pad yaw estimate
    and propagate covariance properly. While the yaw estimate is poor (e.g. seeding),
    the UKF falls back to the previous yaw-agnostic 3D-box fit on the same data.

    /landing_pad/yolo_car_meas (std_msgs/Float64MultiArray) layout:
        [0:2]   image stamp (sec, nanosec)
        [2:6]   undistorted box edges u_min, u_max, v_min, v_max (px, pinhole K)
        [6]     confidence
        [7]     drone altitude at the image stamp (m)
        [8:12]  intrinsics fx, fy, cx, cy (px)
        [12:15] camera position in the drone-relative local-level frame (m)
        [15:24] camera rotation (level <- camera), row-major
        [24:28] pipeline timestamps (s, node clock): image received, processing
                started, inference done, measurement published. NaN when
                diagnostics_enabled is False.

    Carrying the camera pose (attitude + gimbal angle, both interpolated to the image
    stamp) in the measurement lets the UKF replay it exactly during OOSM rollback.
    Carrying the pipeline timestamps in the same message means the controller never has
    to pair the measurement with a separate timing message.

    When several cars are detected, the node keeps following the same one: it picks the
    candidate nearest the previously chosen box, and only falls back to the most
    confident detection once that track has gone stale (see _select_candidate).

    This node does not command the gimbal servo. It publishes the car image target on
    /landing_pad/yolo_target for gimbal_controller.py, which owns the actuator
    independently, so YOLO can continue to drive the gimbal even when the AprilTag node
    is disabled.

    Split out from the combined vision_perception node into its own process so a slow
    YOLO inference tick (OpenVINO on CPU) can never stall apriltag.py's detector or the
    independent gimbal controller.
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
        self.declare_parameter("yolo_car_class_id", 0)
        self._car_class_id = (self.get_parameter("yolo_car_class_id").get_parameter_value().integer_value)

        self.declare_parameter("yolo_edge_margin_px", 10)
        self._yolo_edge_margin_px = (self.get_parameter("yolo_edge_margin_px").get_parameter_value().integer_value)
        self.declare_parameter("yolo_ground_ray_z_epsilon", 1e-3)
        self._ground_ray_z_epsilon = (self.get_parameter("yolo_ground_ray_z_epsilon").get_parameter_value().double_value)

        # ---- TRACK CONTINUITY PARAMETERS ----
        # Max jump of the box centre between frames, in box sizes, to count as the same car
        self.declare_parameter("yolo_track_gate", 1.0)
        self._track_gate = (self.get_parameter("yolo_track_gate").get_parameter_value().double_value)
        # After this long without a matching detection, reacquire the most confident car
        self.declare_parameter("yolo_track_hold_s", 0.5)
        self._track_hold_s = (self.get_parameter("yolo_track_hold_s").get_parameter_value().double_value)

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
            f"input={self._image_width}x{self._image_height}, car_id={self._car_class_id}"
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
            tuple(camera_distortion_coeffs), # type:ignore
        )
        K = self._intrinsics.matrix
        self._intrinsics_flat = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        self._dist_coeffs = self._intrinsics.dist_coeffs_array

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
        self._buf_lock = threading.Lock()

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

        _img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Three callback groups, one executor thread each (see main()):
        #   image  - stamps frames on true arrival, even mid-inference
        #   timer  - the YOLO inference loop (below)
        #   default - odometry, gimbal angle and the sim /clock subscription
        # Keeping the timer out of the default group stops a saturated inference loop
        # from starving odometry/gimbal updates (stale camera pose paired with fresh
        # boxes) and freezing sim time. _buf_lock guards the two buffers, which are now
        # written and read from different threads.
        self._image_cb_group = MutuallyExclusiveCallbackGroup()
        self.image_subscription = self.create_subscription(
            Image, "/camera/image_raw", self._image_store_callback, _img_qos,
            callback_group=self._image_cb_group,
        )
        self.get_logger().info(
            "YOLO node started, waiting for /camera/image_raw (published directly, "
            "or republished by apriltag.py when it's running in webcam mode)..."
        )

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._landing_pad_found_publisher = self.create_publisher(Bool, "/landing_pad/found", 10)
        self._yolo_target_publisher = self.create_publisher(Vector3Stamped, "/landing_pad/yolo_target", 10)
        self._car_measurement_publisher = self.create_publisher(Float64MultiArray, "/landing_pad/yolo_car_meas", 10)

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
        # (msg, receive time in s, sequence number), swapped in as ONE tuple so the
        # timer can never pair a frame with another frame's receive time
        self._latest_image = None
        self._img_seq = 0              # written only by the image callback
        self._last_processed_seq = -1
        self._debug_frame = None       # handed to the main thread for display

        # ---- TRACK STATE ----
        self._track_box = None        # (cx, cy, w, h) of the last chosen detection
        self._track_time = -np.inf    # image stamp (s) of the last chosen detection

        # ---- PROCESSING TIMER ----
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()
        self._yolo_timer = self.create_timer(
            1.0 / self._yolo_processing_rate, self._yolo_timer_callback,
            callback_group=self._timer_cb_group,
        )

    def _now_sec(self) -> float:
        """ Node clock (sim time in SITL) in seconds. """
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- RECEPTION CALLBACKS ----
    def _image_store_callback(self, msg) -> None:
        """ Store the latest incoming image message with its receive time.

        :param msg: Incoming Image message from /camera/image_raw
        """
        t_rx = self._now_sec()
        self._img_seq += 1
        self._latest_image = (msg, t_rx, self._img_seq)

    def _quad_odometry_callback(self, msg: Odometry) -> None:
        """ Obtain FCU Odometry, buffering stamped attitude/altitude for
            time-correct lookups.

        :param msg: Incoming Odometry message
        """
        with self._buf_lock:
            self._odom_buffer.push(msg)

    def _gimbal_angle_callback(self, msg: Vector3Stamped) -> None:
        """ Buffer the measured gimbal pitch for image-timestamp alignment.

        :param msg: Vector3Stamped from gimbal_controller.py; vector.x is the actual
                    gimbal pitch in degrees
        """
        with self._buf_lock:
            self._gimbal_buffer.push(stamp_to_sec(msg.header.stamp), msg.vector.x)

    def _get_new_image(self):
        """ Take the latest image, but only if it's newer than the last one this node
            already processed.

        :return: (image msg, receive time in s) if a new image is available, else None
        """
        latest = self._latest_image
        if latest is None or latest[2] == self._last_processed_seq:
            return None

        img_msg, t_rx, seq = latest
        self._last_processed_seq = seq
        return img_msg, t_rx

    def _to_frame(self, img_msg):
        """ Convert an Image message to a BGR frame at the processing resolution. """
        frame = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        if frame.shape[0] != self._image_height or frame.shape[1] != self._image_width:
            frame = cv2.resize(
                frame,
                (self._image_width, self._image_height),
                interpolation=cv2.INTER_AREA,
            )

        return frame

    def _yolo_timer_callback(self) -> None:
        """ Fires at yolo_processing_rate. Runs the YOLO car detector independently of
            the AprilTag pipeline, publishes the raw box + image-time camera geometry as
            the UKF measurement, and an image target for the independent gimbal
            controller.

            Pipeline timestamps carried in the measurement ([24:28], node clock):
                t_rx     image callback ran (true arrival, separate callback group)
                t_start  this timer picked the image up       (queue = t_start - t_rx)
                t_inf    detector returned       (inference, incl. decode/resize/pose)
                t_pub    measurement about to be published    (post-processing)
        """

        if not self.yolo_enabled or self.yolo_model is None:
            return

        latest = self._get_new_image()
        if latest is None:
            return  # no new frame since this node last ran
        img_msg, t_rx = latest
        t_start = self._now_sec()

        frame = self._to_frame(img_msg)
        stamp = img_msg.header.stamp

        with self._buf_lock:
            odom = self._odom_buffer.get_at(stamp)
            gimbal_deg = self._gimbal_buffer.get_at(stamp)

        if odom is None:
            self.get_logger().warn(
                "No odometry buffered yet - skipping frame", throttle_duration_sec=1.0
            )
            return
        q, altitude = odom

        if gimbal_deg is None:
            self.get_logger().warn(
                "No gimbal angle buffered yet - skipping frame",
                throttle_duration_sec=1.0,
            )
            return

        # Camera pose in the local-level frame at the image timestamp (attitude and
        # gimbal angle both interpolated to `stamp`).
        T_local_cam = camera_pose_in_level_frame(q, gimbal_deg, self._camera_mount)

        detections = self._yolo_detection(frame)
        t_inf = self._now_sec()

        # Keep following the same car if YOLO returns several candidates
        car_detection = self._select_candidate(detections, stamp_to_sec(stamp))

        if car_detection is not None:
            cx, cy, bw, bh, confidence = car_detection
            u0, u1, v0, v1 = cx - 0.5 * bw, cx + 0.5 * bw, cy - 0.5 * bh, cy + 0.5 * bh

            # Undistort a 3x3 grid over the box (corners, edge midpoints, centre) and
            # take its extent: distortion bends straight edges, so corners alone can
            # miss the true extent. P=K keeps the output in the same pinhole pixel frame
            # the UKF projects into.
            if np.any(self._dist_coeffs):
                us, vs = np.meshgrid([u0, cx, u1], [v0, cy, v1])
                und = cv2.undistortPoints(
                    np.c_[us.ravel(), vs.ravel()].reshape(-1, 1, 2),
                    self._intrinsics.matrix, self._dist_coeffs, P=self._intrinsics.matrix,
                ).reshape(-1, 2)
                u0, u1 = und[:, 0].min(), und[:, 0].max()
                v0, v1 = und[:, 1].min(), und[:, 1].max()
                cx, cy = und[4]   # grid centre

            self._yolo_target_publisher.publish(
                image_target_message(stamp, cx, cy, self._intrinsics, "yolo_target")
            )

            # Only a box whose centre ray points at the ground is a usable measurement.
            # Note: unlike apriltag.py, this only ever publishes True - a missed YOLO
            # detection shouldn't override AprilTag's own found signal on the topic.
            ray = T_local_cam[:3, :3] @ self._intrinsics.matrix_inv @ np.array([cx, cy, 1.0])
            if ray[2] < -self._ground_ray_z_epsilon * np.linalg.norm(ray):
                self._landing_pad_found_publisher.publish(Bool(data=True))
                timing = ([t_rx, t_start, t_inf, self._now_sec()]
                          if self.diagnostics_enabled else [np.nan] * 4)
                meas = Float64MultiArray()
                meas.data = np.r_[
                    stamp.sec, stamp.nanosec, #type: ignore
                    u0, u1, v0, v1,
                    confidence, altitude,
                    self._intrinsics_flat,
                    T_local_cam[:3, 3], T_local_cam[:3, :3].ravel(),
                    timing,
                ].astype(float).tolist()
                self._car_measurement_publisher.publish(meas)

        if self.show_debug_window:
            self._show_yolo_debug(frame, car_detection)
            self._debug_frame = frame   # shown by the main thread (see show_debug_frame)

        if self.save_frames or self.create_video:
            self._frame_recorder.save(frame)

    def show_debug_frame(self) -> None:
        """ Display the latest debug frame. Call from the main thread only: OpenCV
            windows must be driven from one thread, and the timer can run on any
            executor thread.
        """
        frame, self._debug_frame = self._debug_frame, None
        if frame is not None:
            cv2.imshow("YOLO Debug", frame)
        cv2.waitKey(1)

    def _yolo_detection(self, frame):
        """ Run YOLO and return every car detection that passes the confidence, class
            and edge-margin filters.

        :return: list of (cx, cy, width, height, confidence), empty if no car was found
        """
        candidates = []

        results = self.yolo_model(  #type: ignore
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
                if confidence < self.yolo_conf_threshold or int(box.cls[0]) != self._car_class_id:
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

                candidates.append((
                    float((x1 + x2) * 0.5), float((y1 + y2) * 0.5),
                    float(x2 - x1), float(y2 - y1), confidence,
                ))

        return candidates

    def _select_candidate(self, candidates, t):
        """ Keep following the same car across frames. With a fresh track, pick the
            candidate whose centre is nearest the last chosen box (in box sizes). If
            none is within the gate, skip the frame rather than jump to a different
            object. With no recent track (start-up, or no match for track_hold_s), take
            the most confident candidate.

        :param candidates: list of (cx, cy, width, height, confidence)
        :param t:          image stamp (s)
        :return: the chosen (cx, cy, width, height, confidence), or None
        """
        if not candidates:
            return None

        track_fresh = self._track_box is not None and t - self._track_time <= self._track_hold_s

        if not track_fresh:
            best = max(candidates, key=lambda c: c[4])
        else:
            tcx, tcy, tbw, tbh = self._track_box  #type: ignore
            scale = max(tbw, tbh, 10.0)
            dist = [np.hypot(c[0] - tcx, c[1] - tcy) / scale for c in candidates]
            k = int(np.argmin(dist))
            if dist[k] > self._track_gate:
                return None
            best = candidates[k]

        self._track_box = best[:4]
        self._track_time = t
        return best

    def _show_yolo_debug(self, frame, detection):
        """ Draw the YOLO car detection on the debug frame. """
        if detection is None:
            return

        cx, cy, bw, bh, confidence = detection
        box_color = (0, 255, 255)

        x1 = int(cx - bw * 0.5)
        y1 = int(cy - bh * 0.5)
        x2 = int(cx + bw * 0.5)
        y2 = int(cy + bh * 0.5)

        # Bounding box, centre point and crosshair
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, box_color, -1)
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
            f"Car {confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )


# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    # Three threads, one per callback group: image, YOLO timer, and the default group
    # (odometry/gimbal/clock), so inference never blocks pose or sim-time updates.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

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
        if node.show_debug_window:
            # Dispatch callbacks from here so the debug window stays on the main thread
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
                node.show_debug_frame()
        else:
            executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally:
        executor.shutdown()
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