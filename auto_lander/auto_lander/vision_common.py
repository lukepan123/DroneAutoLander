import os
import bisect
import cv2
import numpy as np
import numpy.typing as npt
import tf_transformations

from geometry_msgs.msg import Vector3Stamped

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

""" Shared utilities for the AprilTag (apriltag.py) and YOLO (yolo.py) landing-pad
    perception nodes, and the gimbal controller (gimbal_controller.py).

    These used to live inside one combined vision_perception node/thread. They are
    split out here so the nodes can independently subscribe to the camera and
    odometry topics and run on their own timers without resource contention on a
    shared executor thread - a slow YOLO inference tick can no longer stall
    AprilTag's detector.

    Everything in this module that defines a GEOMETRIC CONVENTION (camera intrinsics,
    camera->level-frame pose, image-target-to-angle-error) must be shared verbatim by
    every node. Because they are separate processes, importing the same function is
    what stops them drifting apart on camera mount/gimbal offsets.

    The perception nodes publish image targets (image_target_message); the gimbal
    controller owns the only actuator, picks which target source has priority
    (LatestValueBuffer) and converts it with image_target_to_angle_error.

    Camera pose: no node broadcasts a camera TF. Each perception node buffers FCU
    odometry (OdometryBuffer) and the measured gimbal angle (GimbalAngleBuffer, fed from
    /landing_pad/gimbal_angle, stamped at measurement time by gimbal_controller.py),
    interpolates both to the image's header.stamp, and calls camera_pose_in_level_frame
    with the mount from load_camera_mount. This is time-aligned to the image and cannot
    fail with a TF extrapolation error - the buffers clamp to the newest sample instead.

    Camera model: the SIYI A2 mini is a 160 deg (horizontal) fisheye, used with its
    on-board distortion correction OFF. Everything works in raw fisheye pixels through
    the Kannala-Brandt model (OpenCV cv2.fisheye convention), see fisheye_project /
    fisheye_pixel_to_ray. The UKF (ukf.py) imports the same two functions, so the
    perception nodes and the filter cannot disagree about the lens.
"""


# ---- FISHEYE CAMERA MODEL ----
# Kannala-Brandt (cv2.fisheye): a ray at angle theta from the optical axis lands at
#     theta_d = theta * (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
#     (u, v) = (fx * theta_d * x / r + cx,  fy * theta_d * y / r + cy),  r = hypot(x, y)
# D = (k1, k2, k3, k4) = 0 is the ideal equidistant lens (r = f theta), the model the
# Gazebo wide-angle camera renders. theta comes from atan2(r, z), not atan(r / z) as in
# cv2.fisheye.projectPoints, so rays beyond 90 deg (and behind the camera) still map to
# a finite, monotonic radius: they land outside the image and get clipped there.
# Camera frame: x right, y down, z forward (OpenCV).

def fisheye_project(pts_cam: np.ndarray, K: np.ndarray, D) -> np.ndarray:
    """ Camera-frame points (..., 3) -> fisheye pixels (..., 2).

    :param pts_cam: Points in the camera frame, any leading shape
    :param K:       3x3 camera matrix
    :param D:       Fisheye coefficients (k1, k2, k3, k4)
    :return: Pixel coordinates, same leading shape as pts_cam
    """
    x, y, z = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]
    k1, k2, k3, k4 = D
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    th2 = th * th
    th_d = th * (1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
    # theta_d / r -> 1 / z on the optical axis
    on_axis = r < 1e-12
    scale = np.where(on_axis, 1.0 / np.maximum(z, 1e-12), th_d / np.where(on_axis, 1.0, r))
    return np.stack([K[0, 0] * x * scale + K[0, 2],
                     K[1, 1] * y * scale + K[1, 2]], axis=-1)


def fisheye_pixel_to_ray(u: float, v: float, K: np.ndarray, D) -> np.ndarray:
    """ Fisheye pixel -> unit ray in the camera frame (3,). Inverse of fisheye_project,
        valid beyond 90 deg.

    :param u, v: Pixel coordinates
    :param K:    3x3 camera matrix
    :param D:    Fisheye coefficients (k1, k2, k3, k4)
    :return: Unit ray (x right, y down, z forward)
    """
    mx = (u - K[0, 2]) / K[0, 0]
    my = (v - K[1, 2]) / K[1, 1]
    th_d = np.hypot(mx, my)
    if th_d < 1e-12:
        return np.array([0.0, 0.0, 1.0])

    # Invert theta_d(theta) with Newton; theta_d is monotonic for a sane calibration
    k1, k2, k3, k4 = D
    th = th_d
    for _ in range(10):
        th2 = th * th
        f = th * (1 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4)))) - th_d
        df = 1 + th2 * (3 * k1 + th2 * (5 * k2 + th2 * (7 * k3 + th2 * 9 * k4)))
        step = f / df
        th -= step
        if abs(step) < 1e-12:
            break

    s = np.sin(th) / th_d
    return np.array([mx * s, my * s, np.cos(th)])


def get_workspace_root() -> str | None:
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


def stamp_to_sec(stamp) -> float:
    """ builtin_interfaces/Time -> float seconds. """
    return stamp.sec + stamp.nanosec * 1e-9


def image_target_message(stamp, centre_x: float, centre_y: float, intrinsics, frame_id: str):
    """ Pack an image target and the frame's vertical FOV into a ROS message.

        vector.x = horizontal target coordinate, 0..1 across the image
        vector.y = vertical target coordinate, 0..1 down the image
        vector.z = image vertical FOV (radians)

        The fisheye pixel is turned into its ray and encoded as a tangent, so that
        image_target_to_angle_error (atan of a tangent, unchanged in the gimbal
        controller) returns the target's true angle. With fov_vertical = the true angle
        of the top/bottom edge, the image edges still encode to exactly 0 and 1.
    """
    ray = fisheye_pixel_to_ray(centre_x, centre_y, intrinsics.matrix, intrinsics.dist_coeffs_array)
    z = max(ray[2], 1e-6)
    msg = Vector3Stamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.vector.x = float(0.5 + (ray[0] / z) / (2.0 * np.tan(intrinsics.fov_horizontal / 2.0)))
    msg.vector.y = float(0.5 + (ray[1] / z) / (2.0 * np.tan(intrinsics.fov_vertical / 2.0)))
    msg.vector.z = float(intrinsics.fov_vertical)
    return msg


def image_target_to_angle_error(normalised_centre_y: float, fov_vertical: float) -> float:
    """ Convert an image target's vertical coordinate into a signed vertical angle error.

        Positive = target below the image centre. The image target carries the
        vertical FOV so this remains valid for the different processing resolutions
        used by AprilTag and YOLO. Exact for the fisheye: see image_target_message.
    """
    return float(
        np.degrees(
            np.arctan2(
                2.0 * (normalised_centre_y - 0.5) * np.tan(fov_vertical / 2.0),
                1.0,
            )
        )
    )


@dataclass
class TagDefinition:
    """ Defines an AprilTag definition (tag size and position on the landing pad). """

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


@dataclass
class CameraIntrinsics:
    """ Fisheye camera model (Kannala-Brandt, see fisheye_project) for the frame as
        processed. Every node builds one from the same img_width/img_height parameters
        so their pixel<->ray math stays consistent.

        Without a calibration (matrix_px all zeros) it is the ideal equidistant lens
        whose horizontal FOV spans the image width: f = (W / 2) / (HFOV / 2), principal
        point at the image centre. That is exactly what the Gazebo wide-angle camera
        renders. For the real camera pass the cv2.fisheye.calibrate result, scaled from
        the calibration resolution to width x height.

        The vertical FOV (true angle between the top and bottom edges) is carried with
        image-target messages so the independent gimbal controller can recover the
        target angle without knowing which node produced it.

    :param dist_coeffs: Fisheye coefficients (k1, k2, k3, k4); zeros = equidistant
    :param matrix_px:   Optional (fx, fy, cx, cy) in px at width x height; all zeros
                        or None = derive from the horizontal FOV
    """

    width: int
    height: int
    fov_horizontal: float
    dist_coeffs: tuple[float, float, float, float]
    matrix_px: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if len(self.dist_coeffs) != 4:
            raise ValueError("fisheye dist_coeffs must hold 4 values (k1, k2, k3, k4)")

        if self.matrix_px is not None and any(self.matrix_px):
            fx, fy, cx, cy = self.matrix_px
        else:
            fx = fy = (self.width / 2) / (self.fov_horizontal / 2)
            cx, cy = self.width / 2, self.height / 2

        self.matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs_array = np.array(self.dist_coeffs, dtype=np.float64)

        # True angle from the optical axis to the top edge, doubled
        top = fisheye_pixel_to_ray(cx, 0.0, self.matrix, self.dist_coeffs_array)
        self.fov_vertical = 2 * np.arctan2(-top[1], top[2])


@dataclass
class CameraMount:
    """ Camera mount translation and fixed Euler-angle offsets in the body frame. """

    offset_x: float
    offset_y: float
    offset_z: float
    roll_offset: float
    pitch_offset: float
    yaw_offset: float


def load_camera_mount(node) -> CameraMount:
    """ Declare and read the camera mount parameters on a node and return them as a
        CameraMount. Both perception nodes call this so the mount defaults live in
        exactly one place and can't drift apart between processes.

    :param node: The rclpy Node declaring the parameters
    :return: CameraMount built from the node's camera_offset_* / camera_mount_* params
    """
    defaults = {
        "camera_offset_x": 0.02,
        "camera_offset_y": -0.01,
        "camera_offset_z": -0.124923,
        "camera_mount_roll": -1.5707963,
        "camera_mount_pitch": 0.0,
        "camera_mount_yaw": -1.5707963,
    }
    v = {
        name: node.declare_parameter(name, default).get_parameter_value().double_value
        for name, default in defaults.items()
    }
    return CameraMount(
        offset_x=v["camera_offset_x"],
        offset_y=v["camera_offset_y"],
        offset_z=v["camera_offset_z"],
        roll_offset=v["camera_mount_roll"],
        pitch_offset=v["camera_mount_pitch"],
        yaw_offset=v["camera_mount_yaw"],
    )


def camera_pose_in_level_frame(
    quad_rotation, servo_angle, camera_mount: CameraMount
) -> np.ndarray:
    """ Camera pose (position + rotation) expressed in the drone-relative,
        local-level frame - the drone's own translation is excluded, only its
        attitude and the gimbal angle are applied. This is the same frame
        convention used for both the AprilTag and YOLO landing-pad measurements.
        Both nodes MUST call this exact function (not a re-implementation) so
        they can't drift apart on camera mount/gimbal geometry now that they
        run as separate processes.

    :param quad_rotation: Quadcopter rotation quaternion [x, y, z, w]
    :param servo_angle:   Gimbal servo angle (degrees)
    :param camera_mount:  Camera body offset and fixed mounting rotation
    :return: 4x4 homogeneous transform: level-frame <- camera-frame
    """

    # Quad_local -> Quad body
    T_quad_local = tf_transformations.quaternion_matrix(quad_rotation)

    # Quad_body -> Cam
    t_quad_cam = np.array(
        [
            camera_mount.offset_x,
            camera_mount.offset_y,
            camera_mount.offset_z,
        ]
    )
    q_quad_cam = tf_transformations.quaternion_from_euler(
        camera_mount.roll_offset + np.deg2rad(servo_angle),
        camera_mount.pitch_offset,
        camera_mount.yaw_offset,
    )
    T_quad_cam = tf_transformations.quaternion_matrix(q_quad_cam)
    T_quad_cam[:3, 3] = t_quad_cam

    return T_quad_local @ T_quad_cam


class TimeInterpolatedBuffer:
    """ Generic (timestamp -> value) ring buffer with pluggable interpolation, used to
        time-align data arriving on one topic (odometry, gimbal angle) with an image
        captured at some other timestamp.
    """

    def __init__(self, window_s: float = 1.0, maxlen: int = 400):
        self._buf: deque[tuple[float, object]] = deque(maxlen=maxlen)
        self._window_s = window_s

    def push(self, t: float, value) -> None:
        self._buf.append((t, value))
        cutoff = t - self._window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    def get_at(self, t_query: float, interp_fn):
        """ Interpolate to t_query using interp_fn(v0, v1, fraction) -> value.
            Clamps to the oldest/newest sample if t_query falls outside the
            buffered window. Returns None if the buffer is empty.
        """
        if not self._buf:
            return None

        times = [t for t, _ in self._buf]

        if t_query <= times[0]:
            return self._buf[0][1]
        if t_query >= times[-1]:
            return self._buf[-1][1]

        idx = bisect.bisect_right(times, t_query)
        t0, v0 = self._buf[idx - 1]
        t1, v1 = self._buf[idx]

        if t1 <= t0:
            return v0

        fraction = (t_query - t0) / (t1 - t0)
        return interp_fn(v0, v1, fraction)


class LatestValueBuffer:
    """ Holds only the most recent (value, timestamp) sample and reports whether it
        is still fresh relative to some query time. Unlike TimeInterpolatedBuffer,
        this is NOT for aligning data to a specific past image timestamp - it's for
        live control-loop target selection where the question is simply "is this
        still current enough to act on right now?" (e.g. gimbal_controller.py deciding
        whether to use the latest YOLO target when an AprilTag target has gone stale).
    """

    def __init__(self):
        self._value = None
        self._t: float | None = None

    def push(self, value, t: float) -> None:
        self._value = value
        self._t = t

    def get_if_fresh(self, t_now: float, max_age_s: float):
        """
        :param t_now:     Current time (seconds) to judge freshness against
        :param max_age_s: Maximum allowed age (seconds) before the sample is
                          considered stale
        :return: The buffered value if it exists and is within max_age_s of t_now,
                 otherwise None
        """
        if self._value is None or self._t is None:
            return None
        if (t_now - self._t) > max_age_s:
            return None
        return self._value


class OdometryBuffer:
    """ Buffers stamped (orientation quaternion, altitude) samples from
        /mavros/global_position/local and interpolates (SLERP for orientation, linear
        for altitude) to an arbitrary query timestamp - typically an image's
        header.stamp. Both nodes maintain their own instance; the odometry topic is
        cheap enough to subscribe to twice that this is simpler and safer than trying
        to share one buffer across two processes.
    """

    def __init__(self, window_s: float = 1.0):
        self._buffer = TimeInterpolatedBuffer(window_s=window_s, maxlen=400)

    def push(self, msg) -> None:
        t = stamp_to_sec(msg.header.stamp)
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        altitude = msg.pose.pose.position.z
        self._buffer.push(t, (q, altitude))

    def get_at(self, stamp) -> tuple[list[float], float] | None:
        """
        :param stamp: builtin_interfaces/Time, e.g. the image's header.stamp
        :return: ([x, y, z, w] quaternion, altitude) at that instant, or None if the
            buffer is empty
        """
        t_query = stamp_to_sec(stamp)

        def interp(v0, v1, fraction):
            q0, alt0 = v0
            q1, alt1 = v1
            slerp_result = cast(
                npt.NDArray[np.floating],
                tf_transformations.quaternion_slerp(q0, q1, fraction),
            )
            altitude = alt0 + fraction * (alt1 - alt0)
            return list(slerp_result), altitude

        result = self._buffer.get_at(t_query, interp)
        if result is None:
            return None
        q, alt = result #type: ignore
        return list(q), alt

    def __len__(self) -> int:
        return len(self._buffer)


class GimbalAngleBuffer:
    """ Buffers the gimbal angle published on /landing_pad/gimbal_angle so it can be
        interpolated to an image's header.stamp. Together with OdometryBuffer this
        replaces the old gimbal camera TF: each perception node feeds both into
        camera_pose_in_level_frame. get_at clamps to the newest/oldest sample rather
        than raising, so a gimbal sample arriving slightly after an image is harmless.
        Simple linear interpolation: the servo range (-90..25 deg) never wraps.
    """

    def __init__(self, window_s: float = 1.0):
        self._buffer = TimeInterpolatedBuffer(window_s=window_s, maxlen=400)

    def push(self, t: float, angle_deg: float) -> None:
        self._buffer.push(t, angle_deg)

    def get_at(self, stamp) -> float | None:
        t_query = stamp_to_sec(stamp)

        def interp(v0, v1, fraction):
            return v0 + fraction * (v1 - v0)

        return self._buffer.get_at(t_query, interp)  #type: ignore

    def __len__(self) -> int:
        return len(self._buffer)


class FrameRecorder:
    """ Handles optional per-frame JPEG saving and/or stitching a debug video from the
        saved frames. Each node owns its own instance with a distinct `tag` (e.g.
        "apriltag" / "yolo") so two nodes writing to the same output_dir at the same
        time don't collide on filenames.
    """

    def __init__(self, logger, tag: str, output_dir: str, video_fps: float,
                 save_frames: bool, create_video: bool):
        self._logger = logger
        self._tag = tag
        self._output_dir = output_dir
        self._video_fps = video_fps
        self._save_frames = save_frames
        self._create_video = create_video
        self._frame_count = 0
        self._saved_frames: list[str] = []
        self.frames_dir = None
        self.video_filename = None

        if self._save_frames or self._create_video:
            self._start()
        else:
            self._logger.info(
                f"[{self._tag}] Frame saving DISABLED - no video will be created"
            )

    def _start(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._output_dir:
            self.frames_dir = os.path.join(
                self._output_dir, f"{self._tag}_frames_{timestamp}"
            )
        else:
            base_dir = get_workspace_root() or os.getcwd()
            self.frames_dir = os.path.join(base_dir, f"{self._tag}_frames_{timestamp}")

        os.makedirs(self.frames_dir, exist_ok=True)
        self._logger.info(
            f"[{self._tag}] Frame saving ENABLED - Directory: {self.frames_dir}"
        )

        self.video_filename = os.path.join(
            os.path.dirname(self.frames_dir),
            f"{self._tag}_detection_video_{timestamp}.mp4",
        )
        self._logger.info(f"[{self._tag}] Video will be saved as: {self.video_filename}")

    def save(self, frame) -> None:
        if self.frames_dir is None:
            self._logger.warning(
                f"[{self._tag}] Frame saving enabled but frames_dir not initialized"
            )
            return

        frame_filename = os.path.join(
            self.frames_dir, f"frame_{self._frame_count:06d}.jpg"
        )
        if cv2.imwrite(frame_filename, frame):
            self._saved_frames.append(frame_filename)
            self._frame_count += 1
            if self._frame_count % 100 == 0:
                self._logger.info(f"[{self._tag}] Saved {self._frame_count} frames so far...")
        else:
            self._logger.warning(f"[{self._tag}] Failed to save frame {self._frame_count}")

    def finalize(self) -> None:
        """ Stitch saved frames into an mp4 (if create_video) and clean up loose frame
            files afterwards (unless save_frames was also requested).
        """
        if not (self._save_frames or self._create_video) or not self._saved_frames:
            self._logger.info(
                f"[{self._tag}] Video creation skipped. "
                f"save_frames={self._save_frames}, create_video={self._create_video}, "
                f"frames_count={len(self._saved_frames)}"
            )
            return

        try:
            duration_seconds = len(self._saved_frames) / self._video_fps
            self._logger.info(
                f"[{self._tag}] Creating video from {len(self._saved_frames)} frames "
                f"(estimated duration: {duration_seconds:.1f}s at {self._video_fps}fps)..."
            )

            first_frame = cv2.imread(self._saved_frames[0])
            if first_frame is None:
                self._logger.error(
                    f"[{self._tag}] Could not read first frame for video creation"
                )
                return

            height, width, _ = first_frame.shape
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, self._video_fps, (width, height)  #type: ignore
            )

            if not video_writer.isOpened():
                self._logger.error(f"[{self._tag}] Failed to open video writer")
                return

            frames_written = 0
            for i, frame_path in enumerate(self._saved_frames):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
                    frames_written += 1
                    if (i + 1) % 100 == 0:
                        self._logger.info(
                            f"[{self._tag}] Writing frame {i + 1}/{len(self._saved_frames)} to video..."
                        )
                else:
                    self._logger.warning(f"[{self._tag}] Could not read frame: {frame_path}")

            video_writer.release()
            self._logger.info(
                f"[{self._tag}] Video created successfully: {self.video_filename}"
            )
            self._logger.info(
                f"[{self._tag}] Final video stats: {frames_written} frames written, "
                f"duration: {frames_written / self._video_fps:.1f}s"
            )

            if not self._save_frames:
                self._logger.info(f"[{self._tag}] Cleaning up temporary frame files...")
                for frame_path in self._saved_frames:
                    try:
                        os.remove(frame_path)
                    except OSError as e:
                        self._logger.warning(
                            f"[{self._tag}] Could not remove frame {frame_path}: {e}"
                        )
                try:
                    os.rmdir(self.frames_dir)  #type: ignore
                except OSError:
                    pass

        except Exception as e:
            self._logger.error(f"[{self._tag}] Error creating video: {e}")