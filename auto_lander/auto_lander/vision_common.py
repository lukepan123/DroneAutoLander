import os
import bisect
import cv2
import numpy as np
import numpy.typing as npt
import tf_transformations

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
    camera->level-frame pose) must be shared verbatim by both nodes. Now that they are
    separate processes, they can no longer drift apart on camera mount/gimbal offsets
    simply because they import the same function.
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
        build one of these from the same imgsz_width/imgsz_height parameters so their
        pixel<->ray math stays consistent.
    """

    width: int
    height: int
    fov_horizontal: float = 2.0  # radians (~114.6 deg) - tune for your camera

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
        self.dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)


def camera_pose_in_level_frame(quad_rotation, servo_angle) -> np.ndarray:
    """ Camera pose (position + rotation) expressed in the drone-relative,
        local-level frame - the drone's own translation is excluded, only its
        attitude and the gimbal angle are applied. This is the same frame
        convention used for both the AprilTag and YOLO landing-pad transforms.
        Both nodes MUST call this exact function (not a re-implementation) so
        they can't drift apart on camera mount/gimbal geometry now that they
        run as separate processes.

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


class TimeInterpolatedBuffer:
    """ Generic (timestamp -> value) ring buffer with pluggable interpolation, used to
        time-align data arriving on one topic (odometry, gimbal angle) with an image
        captured at some other timestamp. Both nodes need this same alignment
        behaviour against their own image stamps, so it lives here once rather than
        being duplicated (and potentially drifting) in each node.
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
    """ Buffers the AprilTag node's published gimbal servo angle so the YOLO node -
        which does not itself drive the gimbal now that the two pipelines are separate
        processes - can recover the angle that was actually in effect when its own
        frame was captured, for the back-projection in _estimate_yolo_ground_position.
        Simple linear interpolation: the servo range (-135..45 deg) never wraps.
    """

    def __init__(self, window_s: float = 1.0):
        self._buffer = TimeInterpolatedBuffer(window_s=window_s, maxlen=400)

    def push(self, t: float, angle_deg: float) -> None:
        self._buffer.push(t, angle_deg)

    def get_at(self, stamp) -> float | None:
        t_query = stamp_to_sec(stamp)

        def interp(v0, v1, fraction):
            return v0 + fraction * (v1 - v0)

        return self._buffer.get_at(t_query, interp) #type: ignore

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
                self._logger.error(f"[{self._tag}] Could not read first frame for video creation")
                return

            height, width, _ = first_frame.shape
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, self._video_fps, (width, height) #type: ignore
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
            self._logger.info(f"[{self._tag}] Video created successfully: {self.video_filename}")
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
                        self._logger.warning(f"[{self._tag}] Could not remove frame {frame_path}: {e}")
                try:
                    os.rmdir(self.frames_dir) #type: ignore
                except OSError:
                    pass

        except Exception as e:
            self._logger.error(f"[{self._tag}] Error creating video: {e}")
