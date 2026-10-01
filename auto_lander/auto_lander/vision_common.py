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
    perception nodes.

    These used to live inside one combined vision_perception node/thread. They are
    split out here so both nodes can independently subscribe to the camera and
    odometry topics, run on their own timers, and broadcast their own TF without
    resource contention on a shared executor thread - a slow YOLO inference tick can
    no longer stall AprilTag's gimbal control / TF broadcast loop.

    Everything in this module that defines a GEOMETRIC CONVENTION (camera intrinsics,
    camera->level-frame pose, pixel-row-to-angle-error) must be shared verbatim by
    both nodes. Now that they are separate processes, they can no longer drift apart
    on camera mount/gimbal offsets simply because they import the same function.

    pixel_row_to_angle_error and LatestValueBuffer provide the shared image geometry
    and timestamped-value behaviour used by the independent gimbal controller. The
    perception nodes publish image targets, while gimbal_controller.py owns the only
    actuator and decides which target source has priority.

    Camera pose: gimbal_controller.py does NOT broadcast a camera TF. Each perception
    node instead buffers FCU odometry (OdometryBuffer) and the measured gimbal angle
    (GimbalAngleBuffer, fed from /landing_pad/gimbal_angle), interpolates both to the
    image's header.stamp, and calls camera_pose_in_level_frame with the mount from
    load_camera_mount. This is time-aligned to the image and cannot fail with a TF
    extrapolation error - the buffers clamp to the newest sample instead.
"""


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


def pixel_row_to_angle_error(centre_y: float, image_height: int, fy: float) -> float:
    """ Convert a detection's vertical pixel centre into a signed angle (degrees)
        off the image's optical-axis row. This is the exact calculation
        apriltag.py's gimbal controller has always used on its own AprilTag corner
        points; it's pulled out here so yolo.py can produce a directly comparable
        error from its own bounding-box centre using its own (different-resolution)
        camera intrinsics, and apriltag.py can treat the two as interchangeable inputs
        to the same PD control law. Purely a function of that frame's own image
        geometry - independent of the drone's attitude or the current servo angle.

    :param centre_y: Detection centre row, in pixels of the frame it came from
    :param image_height: Height (pixels) of that same frame
    :param fy: Vertical focal length (pixels) of that same frame's camera intrinsics
    :return: Signed angle error in degrees; positive = detection below image centre
    """
    pixel_error = centre_y - image_height / 2
    return float(np.degrees(np.arctan2(pixel_error, fy)))


def image_target_message(stamp, centre_x: float, centre_y: float, intrinsics, frame_id: str):
    """ Pack a normalised image target and the frame's vertical FOV into a ROS message.

        vector.x = normalised image x coordinate (0..1)
        vector.y = normalised image y coordinate (0..1)
        vector.z = image vertical FOV (radians)
    """
    msg = Vector3Stamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.vector.x = float(centre_x / intrinsics.width)
    msg.vector.y = float(centre_y / intrinsics.height)
    msg.vector.z = float(intrinsics.fov_vertical)
    return msg


def image_target_to_angle_error(normalised_centre_y: float, fov_vertical: float) -> float:
    """ Convert a normalised image-row target into a signed vertical angle error.

        Positive = target below the image centre. The image target carries the
        vertical FOV so this remains valid for the different processing resolutions
        used by AprilTag and YOLO.
    """
    return float(
        np.degrees(
            np.arctan2(
                2.0 * (normalised_centre_y - 0.5) * np.tan(fov_vertical / 2.0),
                1.0,
            )
        )
    )


def transform_stamped_to_matrix(transform: object) -> np.ndarray:
    """ Convert a geometry_msgs TransformStamped into a 4x4 homogeneous matrix. """
    tf_msg = transform.transform  # type: ignore
    q = [
        tf_msg.rotation.x,
        tf_msg.rotation.y,
        tf_msg.rotation.z,
        tf_msg.rotation.w,
    ]
    T = tf_transformations.quaternion_matrix(q)
    T[:3, 3] = np.array(
        [
            tf_msg.translation.x,
            tf_msg.translation.y,
            tf_msg.translation.z,
        ]
    )
    return T


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
    """ Pinhole camera model derived from image size + horizontal FOV. Both nodes
        build one of these from the same img_width/img_height parameters so their
        pixel<->ray math stays consistent.

        Distortion coefficients are passed in from the node parameters so the camera
        calibration can be changed without touching this shared module. The vertical
        FOV is also carried with image-target messages so the independent gimbal
        controller can recover the target angle without knowing which node produced it.
    """

    width: int
    height: int
    fov_horizontal: float
    dist_coeffs: tuple[float, float, float, float, float]

    def __post_init__(self):
        self.fov_vertical = 2 * np.arctan(
            np.tan(self.fov_horizontal / 2) / (self.width / self.height)
        )

        fx = self.width / (2 * np.tan(self.fov_horizontal / 2))
        fy = self.height / (2 * np.tan(self.fov_vertical / 2))
        self.matrix = np.array(
            [
                [fx, 0, self.width / 2],
                [0, fy, self.height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.matrix_inv = np.linalg.inv(self.matrix)
        self.dist_coeffs_array = np.array(self.dist_coeffs, dtype=np.float64)


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
        convention used for both the AprilTag and YOLO landing-pad transforms.
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