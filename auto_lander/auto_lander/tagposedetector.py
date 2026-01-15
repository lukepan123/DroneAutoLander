#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2

import numpy as np
import tf_transformations
from datetime import datetime
import signal
import sys

# Get the workspace root directory
def get_workspace_root():
    """Find the workspace root by looking for colcon workspace structure"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'src')) and \
           os.path.exists(os.path.join(current_dir, 'build')) and \
           os.path.exists(os.path.join(current_dir, 'install')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

# Actual ROS Node
class ArUCoNode(Node):
    def __init__(self):
        super().__init__("yolo_image_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("enable_debug_publish", False)
        self.enable_debug_publish = (
            self.get_parameter("enable_debug_publish").get_parameter_value().bool_value
        )

        self.declare_parameter("image_source", "topic")  # 'topic' or 'webcam'
        self.image_source = (
            self.get_parameter("image_source").get_parameter_value().string_value
        )

        self.declare_parameter("webcam_index", 0)
        self.webcam_index = int(
            self.get_parameter("webcam_index").get_parameter_value().integer_value
        )

        self.declare_parameter("show_debug_window", True)
        self.show_debug_window = (
            self.get_parameter("show_debug_window").get_parameter_value().bool_value
        )

        # Frame saving and video creation parameters
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

        # Log parameter values for debugging
        self.get_logger().info(f"Video recording parameters: save_frames={self.save_frames}, create_video={self.create_video}, video_fps={self.video_fps}")
        self.get_logger().info(f"Output directory: '{self.output_dir}' (empty means workspace root)")


        # Must match the IR input size you exported (default OpenVINO export is 640)
        self.declare_parameter("imgsz", 256)
        self.imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        # ----------------------------------------------------------

        # Publishers
        self.target_pose_publisher = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.bridge = CvBridge()
        self.webcam_publisher = self.create_publisher(Image, "/image", 10)

        # Initialize frame saving
        self.frame_count = 0
        self.saved_frames = []
        
        if self.save_frames or self.create_video:
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.output_dir:
                self.frames_dir = os.path.join(self.output_dir, f"frames_{timestamp}")
            else:
                # Use workspace root or current directory
                workspace_root = get_workspace_root()
                base_dir = workspace_root if workspace_root else os.getcwd()
                self.frames_dir = os.path.join(base_dir, f"frames_{timestamp}")
            
            os.makedirs(self.frames_dir, exist_ok=True)
            self.get_logger().info(f"Frame saving ENABLED - Directory: {self.frames_dir}")
            self.get_logger().info(f"Video creation settings - save_frames: {self.save_frames}, create_video: {self.create_video}, fps: {self.video_fps}")
            
            # Video output filename
            self.video_filename = os.path.join(
                os.path.dirname(self.frames_dir), 
                f"yolo_detection_video_{timestamp}.mp4"
            )
            self.get_logger().info(f"Video will be saved as: {self.video_filename}")
        else:
            self.get_logger().info("Frame saving DISABLED - no video will be created")

        if self.show_debug_window:
            cv2.namedWindow("Detected Markers", cv2.WINDOW_AUTOSIZE)

        # State
        self.image_width = self.imgsz
        self.image_height = self.imgsz
        self.camera_fov_horizontal = 0.8  # radians (≈114.6°) – tune for your camera
        self.camera_fov_vertical = 0.8  # radians - since square camera

        # Image source
        if self.image_source == "topic":
            self.image_subscription = self.create_subscription(
                Image, "/camera/image_raw", self.image_callback, 10
            )
            self.get_logger().info(
                "ArUCoImageNode started in TOPIC mode, waiting for MAVROS altitude and image topic..."
            )
        else:
            # Try different backends for camera access
            self.cap = None
            backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends_to_try:
                try:
                    self.cap = cv2.VideoCapture(self.webcam_index, backend)
                    if self.cap.isOpened():
                        self.get_logger().info(f"Successfully opened camera {self.webcam_index} with backend {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    self.get_logger().warning(f"Failed to open camera with backend {backend}: {e}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None

            if self.cap is None or not self.cap.isOpened():
                self.get_logger().error(
                    f"Could not open webcam at index {self.webcam_index}. "
                    f"Make sure your user is in the 'video' group: sudo usermod -a -G video $USER"
                )
            else:
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.imgsz)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imgsz)
                
                # Log actual camera properties
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                self.get_logger().info(
                    f"ArUCoImageNode started in WEBCAM mode. Camera properties: "
                    f"FPS={actual_fps}, Width={actual_width}, Height={actual_height}"
                )
            self.timer = self.create_timer(
                1.0 / 30.0, self.webcam_timer_callback
            )  # 30 Hz

    def webcam_timer_callback(self):
        """Read image from webcam and publish to /image topic"""
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to IR input size (square) – must match export
                frame = cv2.resize(
                    frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST
                )
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

                self.image_callback(msg)
            else:
                self.get_logger().warning("Failed to read frame from webcam.")
        else:
            self.get_logger().warning("Webcam not opened.")

    def altitude_callback(self, msg):
        """Get current quadcopter height from MAVROS altimeter"""
        self.current_altitude = msg.data

    def image_callback(self, msg):
        """Process image and detect tag, calculate altitude"""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Ensure inference size matches IR (handles topic frames of any size)
        if frame.shape[0] != self.imgsz or frame.shape[1] != self.imgsz:
            frame = cv2.resize(
                frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST
            )

        # Inference (ArUCo detection via OpenCV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Change to greyscale before inference step
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray_frame)

        # Determine the pose of each marker
        if ids is not None:
            fx = self.image_width / (2 * np.tan(self.camera_fov_horizontal / 2))
            fy = self.image_height / (2 * np.tan(self.camera_fov_vertical / 2))
            camera_matrix = np.array([
                [fx, 0, self.image_width/2],
                [0, fy, self.image_height/2 ],
                [0, 0, 1],
            ], dtype=np.float64)

            dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)

            TAG_SIZES = {
                35: 0.54,
                27: 0.108,
            }

            for i, id in enumerate(ids):
                tag_id = int(id[0])

                if tag_id not in TAG_SIZES:
                    continue

                marker_length = TAG_SIZES[tag_id]
                half = marker_length / 2.0

                object_points = np.array([
                    [ half,  half, 0],  # top-right
                    [ half, -half, 0],  # bottom-right
                    [-half, -half, 0],  # bottom-left
                    [-half,  half, 0],  # top-left
                ], dtype=np.float32)

                image_points = corners[i][0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Draw pose axes for debugging
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        marker_length * 0.5
                    )

                    # Convert camera pose to a target pose (camera coords to world coords)
                    pose_msg = self.rvec_tvec_to_posestamped(
                        rvec,
                        tvec,
                        frame_id='quadcopter'
                    )

                    self.target_pose_publisher.publish(pose_msg)
                    print(pose_msg)

        # Show the output image after ArUCo detection (if debug window enabled)
        if self.show_debug_window:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow('Detected Markers', frame)
            cv2.waitKey(1)

        # Save frame if enabled
        if self.save_frames or self.create_video:
            if hasattr(self, 'frames_dir'):
                frame_filename = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.jpg")
                success = cv2.imwrite(frame_filename, frame)
                if success:
                    self.saved_frames.append(frame_filename)
                    self.frame_count += 1
                    
                    # Log progress every 100 frames
                    if self.frame_count % 100 == 0:
                        self.get_logger().info(f"Saved {self.frame_count} frames so far...")
                else:
                    self.get_logger().warning(f"Failed to save frame {self.frame_count}")
            else:
                self.get_logger().warning("Frame saving enabled but frames_dir not initialized")

        if self.enable_debug_publish:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.webcam_publisher.publish(msg)
            
    def rvec_tvec_to_posestamped(self, rvec, tvec, frame_id='quadcopter'):
        pose = PoseStamped()

        # Camera relative to quad - position vector and rotation matrix (m)
        pos_cam_to_quad = np.array([0.0, 0.0, 0.7])
        R_cam_to_quad = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ])

        pos_cam = tvec.reshape(3) # Ensure correct shape
        pos_target = R_cam_to_quad @ pos_cam + pos_cam_to_quad

        # Header
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = frame_id

        # Translation (camera frame, meters)
        pose.pose.position.x = float(pos_target[0])
        pose.pose.position.y = float(pos_target[1])
        pose.pose.position.z = float(pos_target[2])

        # Rodrigues to rotation matrix
        R_cam, _ = cv2.Rodrigues(rvec)

        # Rotate into quadcopter frame
        R_target = R_cam_to_quad @ R_cam

        # Build 4x4 homogeneous transform
        T = np.eye(4)
        T[:3, :3] = R_target

        # Rotation matrix → quaternion
        q = tf_transformations.quaternion_from_matrix(T)

        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        return pose

    def create_video_from_frames(self):
        """Create video from saved frames"""
        if not (self.save_frames or self.create_video) or not self.saved_frames:
            self.get_logger().info(f"Video creation skipped. save_frames={self.save_frames}, create_video={self.create_video}, frames_count={len(self.saved_frames) if hasattr(self, 'saved_frames') else 0}")
            return
            
        try:
            duration_seconds = len(self.saved_frames) / self.video_fps
            self.get_logger().info(f"Creating video from {len(self.saved_frames)} frames (estimated duration: {duration_seconds:.1f}s at {self.video_fps}fps)...")
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(self.saved_frames[0])
            if first_frame is None:
                self.get_logger().error("Could not read first frame for video creation")
                return
                
            height, width, layers = first_frame.shape
            self.get_logger().info(f"Video dimensions: {width}x{height}")
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                self.video_filename, 
                fourcc, 
                self.video_fps, 
                (width, height)
            )
            
            if not video_writer.isOpened():
                self.get_logger().error("Failed to open video writer")
                return
            
            # Write all frames to video
            frames_written = 0
            for i, frame_path in enumerate(self.saved_frames):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
                    frames_written += 1
                    
                    # Progress update every 100 frames
                    if (i + 1) % 100 == 0:
                        self.get_logger().info(f"Writing frame {i + 1}/{len(self.saved_frames)} to video...")
                else:
                    self.get_logger().warning(f"Could not read frame: {frame_path}")
            
            video_writer.release()
            self.get_logger().info(f"Video created successfully: {self.video_filename}")
            self.get_logger().info(f"Final video stats: {frames_written} frames written, duration: {frames_written/self.video_fps:.1f}s")
            
            # Optionally clean up frame files
            if not self.save_frames:  # Only delete frames if we don't want to keep them
                self.get_logger().info("Cleaning up temporary frame files...")
                for frame_path in self.saved_frames:
                    try:
                        os.remove(frame_path)
                    except OSError as e:
                        self.get_logger().warning(f"Could not remove frame {frame_path}: {e}")
                        
                # Remove frames directory if empty
                try:
                    os.rmdir(self.frames_dir)
                except OSError:
                    pass  # Directory not empty or other error
                    
        except Exception as e:
            self.get_logger().error(f"Error creating video: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArUCoNode()
    
    # Global variable to track if we're already shutting down
    shutdown_in_progress = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            node.get_logger().warn("Second interrupt received! Force terminating without video creation...")
            sys.exit(1)
        else:
            shutdown_in_progress = True
            node.get_logger().info("Interrupt received, creating video before shutdown (press Ctrl+C again to force quit)...")
            raise KeyboardInterrupt()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally:
        # Create video from saved frames before cleanup
        if hasattr(node, 'create_video_from_frames') and not shutdown_in_progress:
            node.create_video_from_frames()
        elif hasattr(node, 'create_video_from_frames'):
            try:
                # Give it a chance even if shutdown is in progress
                node.get_logger().info("Creating video during shutdown...")
                node.create_video_from_frames()
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
