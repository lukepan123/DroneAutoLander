#!/usr/bin/env python3
import os

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistWithCovarianceStamped
import tf_transformations

from cv_bridge import CvBridge
import cv2

import numpy as np
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


class ArUCoNode(Node):
    def __init__(self):
        super().__init__("yolo_image_node")

        # ---------------- PARAMETERS ----------------
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

        # SHow the camera viewport
        self.declare_parameter("show_debug_window", True)
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

        # Decalre video/frames output directory location
        self.declare_parameter("output_dir", "")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        # Log parameter values for debugging
        self.get_logger().info(f"Video recording parameters: save_frames={self.save_frames}, create_video={self.create_video}, video_fps={self.video_fps}")
        self.get_logger().info(f"Output directory: '{self.output_dir}' (empty means workspace root)")

        # Must match the IR input size you exported (default IRIS is 640 x 480)
        self.declare_parameter("imgsz_width", 640)
        self.image_width = int(self.get_parameter("imgsz_width").get_parameter_value().integer_value)
        
        self.declare_parameter("imgsz_height", 480)
        self.image_height = int(self.get_parameter("imgsz_height").get_parameter_value().integer_value)

        self.camera_fov_horizontal = 0.8  # radians (≈114.6°) – tune for your camera
        self.camera_fov_vertical = 2 * np.arctan(np.tan(self.camera_fov_horizontal / 2) / (self.image_width/self.image_height))
        
        
        # ---------------- PUBLISHERS AND SUBSCRIPTIONS ----------------
        # Initialise MAVROS subscription
        self.pose = PoseStamped()
        self.old_target_pose = PoseWithCovarianceStamped()
        pose_qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, pose_qos)

        # Publishers
        self.target_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/target_pose', 10)
        self.target_twist_publisher = self.create_publisher(TwistWithCovarianceStamped, '/target_twist', 10)
        self.bridge = CvBridge()
        self.webcam_publisher = self.create_publisher(Image, "/image", 10)


        # ---------------- INITIALISATION ----------------
        # AruCo Detector Params
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # self.aruco_params.cornerRefinementWinSize = 5
        # self.aruco_params.cornerRefinementMaxIterations = 30
        # self.aruco_params.cornerRefinementMinAccuracy = 0.01
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Target Velocity state variables
        self.prev_gray = None
        self.prev_corners = None
        self.prev_stamp = None

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
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
                
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


    # ---------------- SUBSCRIPTION FUNCTION CALLBACKS ----------------
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
    
    def local_pose_callback(self, msg: PoseStamped):
        self.pose = msg
    
    def altitude_callback(self, msg):
        """Get current quadcopter height from MAVROS altimeter"""
        self.current_altitude = msg.data

    def image_callback(self, msg):
        """Process image and detect tag, calculate pose and twist"""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Ensure inference size matches IR (handles topic frames of any size)
        if frame.shape[0] != self.image_height or frame.shape[1] != self.image_width:
            frame = cv2.resize(
                frame, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR
            )

        # Inference (ArUCo detection via OpenCV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Change to greyscale before inference step

        # Detect the markers
        corners, ids, rejected = self.detector.detectMarkers(gray_frame)

        if ids is not None:  
            # Camera properties
            fx = self.image_width / (2 * np.tan(self.camera_fov_horizontal / 2))
            fy = self.image_height / (2 * np.tan(self.camera_fov_vertical / 2))
            camera_matrix = np.array([
                [fx, 0, self.image_width/2],
                [0, fy, self.image_height/2 ],
                [0, 0, 1],
            ], dtype=np.float64)

            dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)

            # Determine the pose of each marker
            TAG_SIZES = {35: 0.489}

            for i, marker_id in enumerate(ids):
                tag_id = int(marker_id[0])

                if tag_id not in TAG_SIZES:
                    continue # ignore tags that we dont expect

                marker_length = TAG_SIZES[tag_id]
                half = marker_length / 2.0

                object_points = np.array([
                    [-half,  half, 0],  # top-left
                    [ half,  half, 0],  # top-right
                    [ half, -half, 0],  # bottom-right
                    [-half, -half, 0],  # bottom-left
                ], dtype=np.float32)

                image_points = corners[i][0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
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
                    pose_msg = self.target_posestamped(rvec, tvec)
                    if pose_msg is not None:
                        self.target_pose_publisher.publish(pose_msg)

                    # Estimate velocity
                    twist_msg = self.target_twiststamped(gray_frame, corners, tvec, camera_matrix)
                    if twist_msg is not None:
                        self.target_twist_publisher.publish(twist_msg)


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


    # ---------------- HELPER FUNCTIONS ----------------        
    def target_posestamped(self, rvec, tvec):
        # To figure out where the target is in the world, we need to the following set of transformations:
        # world -> quad -> camera -> target

        # Header for pose
        target_pose = PoseWithCovarianceStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'map'

        # Check that mavros data is up to date
        now = self.get_clock().now()

        if self.pose is None:
            self.get_logger().warn("No MAVROS pose received yet")
            return None

        pose_time = Time.from_msg(self.pose.header.stamp)
        age = (now - pose_time).nanoseconds * 1e-9  # seconds
        MAX_POSE_AGE = 0.2  # seconds (tune this)

        if age < 0.0:
            # self.get_logger().warn(
            #     f"MAVROS pose is from the future (age={age:.3f}s), skipping"
            # )
            return None

        if age > MAX_POSE_AGE:
            # self.get_logger().warn(
            #     f"MAVROS pose too old (age={age:.3f}s), skipping"
            # )
            return None

        # From ArUCo tag, find the camera --> target transform (4x4)
        T_cam_to_target = np.eye(4)
        T_cam_to_target[:3, 3] = tvec.reshape(3) # position
        R_cam_to_target, _ = cv2.Rodrigues(rvec)
        T_cam_to_target[:3, :3] = R_cam_to_target # rotation

        # From quad setup, we know the quad -> camera transform (4x4)
        T_quad_to_cam = np.eye(4)
        T_quad_to_cam[:3, 3] = np.array([0.0, 0.0, -0.1249]) # position
        T_quad_to_cam[:3, :3] = np.array([
            [ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0, -1]
        ]) # rotation

        # From MAVROS, we know the map -> quad transform (4x4)
        T_map_to_quad = np.eye(4)
        T_map_to_quad[:3, 3] = np.array([self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z]) # position
        q_map_to_quad = [self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w]       
        T_map_to_quad[:3, :3] = tf_transformations.quaternion_matrix(q_map_to_quad)[:3, :3] # rotation

        # Now using all the transforms, we can obtain the map -> target transform
        T_map_to_target = T_map_to_quad @ T_quad_to_cam @ T_cam_to_target

        # Extract translation
        t_map_to_target = T_map_to_target[:3, 3]

        # Clean rotation matrix
        R_map_to_target = T_map_to_target[:3, :3]

        if not np.isfinite(R_map_to_target).all():
            self.get_logger().warn("Rotation matrix contains NaN/Inf, skipping frame")
            return None
        
        U, _, Vt = np.linalg.svd(R_map_to_target)
        R_map_to_target = U @ Vt

        # Ensure proper rotation (det = +1)
        if np.linalg.det(R_map_to_target) < 0:
            U[:, -1] *= -1
            R_map_to_target = U @ Vt

        # Rebuild a clean 4x4 transform
        T_map_to_target[:3, :3] = R_map_to_target
        q_map_to_target = tf_transformations.quaternion_from_matrix(T_map_to_target)

        # Translation (camera frame, meters)
        target_pose.pose.pose.position.x = float(t_map_to_target[0])
        target_pose.pose.pose.position.y = float(t_map_to_target[1])
        target_pose.pose.pose.position.z = float(t_map_to_target[2])
        target_pose.pose.pose.orientation.x = q_map_to_target[0]
        target_pose.pose.pose.orientation.y = q_map_to_target[1]
        target_pose.pose.pose.orientation.z = q_map_to_target[2]
        target_pose.pose.pose.orientation.w = q_map_to_target[3]

        # Covariance
        cov = np.zeros((6, 6))
        # Position variance (meters^2)
        cov[0, 0] = 0.3   # x
        cov[1, 1] = 0.3   # y
        cov[2, 2] = 0.3   # z
        # Orientation variance (rad^2)
        cov[3, 3] = 0.2    # roll
        cov[4, 4] = 0.2    # pitch
        cov[5, 5] = 0.2    # yaw

        target_pose.pose.covariance = cov.flatten().tolist()

        return target_pose
    
    def target_twiststamped(self, gray, corners, tvec, camera_matrix):
        # To figure out how fast the target is travelling, we use an optical flow algorithim on the corners of the ARUCO tags

        # Header for twist
        target_twist = TwistWithCovarianceStamped()
        target_twist.header.stamp = self.get_clock().now().to_msg()
        target_twist.header.frame_id = 'map'

        stamp = Time.from_msg(target_twist.header.stamp)

        # Initialise previous frame states on first run
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_corners = corners
            self.prev_stamp = stamp
            return None

        dt = (stamp - self.prev_stamp).nanoseconds * 1e-9
        if dt < 1e-3:
            return None

        # Stack all corners into Nx1x2 array
        p0 = np.concatenate(self.prev_corners, axis=1).reshape(-1, 1, 2).astype(np.float32)

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if p1 is None:
            return None

        good_old = p0[status == 1]
        good_new = p1[status == 1]

        if len(good_old) < 4:
            return None

        pixel_vel = (good_new - good_old) / dt
        mean_pixel_vel = np.mean(pixel_vel, axis=0).reshape(2)

        print(mean_pixel_vel)

        # Update state
        self.prev_gray = gray
        self.prev_corners = corners
        self.prev_stamp = stamp

        if mean_pixel_vel is not None:
            depth_z = max(0.0, abs(float(tvec[2])))
            vx = mean_pixel_vel[0] * depth_z / camera_matrix[0, 0]
            vy = mean_pixel_vel[1] * depth_z / camera_matrix[1, 1] #negative to correct result out of calcOpticalFlowPyrLK
            v_cam = np.array([vx, vy, 0.0])

            # Check that mavros data is up to date
            now = self.get_clock().now()

            if self.pose is None:
                self.get_logger().warn("No MAVROS pose received yet")
                return None

            pose_time = Time.from_msg(self.pose.header.stamp)
            age = (now - pose_time).nanoseconds * 1e-9  # seconds
            MAX_POSE_AGE = 0.2  # seconds (tune this)

            if age < 0.0:
                # self.get_logger().warn(
                #     f"MAVROS pose is from the future (age={age:.3f}s), skipping"
                # )
                return None

            if age > MAX_POSE_AGE:
                # self.get_logger().warn(
                #     f"MAVROS pose too old (age={age:.3f}s), skipping"
                # )
                return None
        
            # Rotate into map frame
            q_map_to_quad = [self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w]       
            R_map_to_quad = tf_transformations.quaternion_matrix(q_map_to_quad)[:3, :3] # rotation

            R_quad_to_cam = np.array([
                [ 0, -1,  0],
                [-1,  0,  0],
                [ 0,  0, -1]
            ])

            R_map_to_cam = R_map_to_quad @ R_quad_to_cam
            v_map = R_map_to_cam @ v_cam

            target_twist.twist.twist.linear.x = float(v_map[0])
            target_twist.twist.twist.linear.y = float(v_map[1])
            target_twist.twist.twist.linear.z = float(v_map[2])

            print(v_map[1])

            # Covariance tuned for optical flow
            cov = np.zeros((6, 6))
            cov[0, 0] = 0.2
            cov[1, 1] = 0.2
            cov[2, 2] = 0.3
            target_twist.twist.covariance = cov.flatten().tolist()

            return target_twist

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


# ---------------- MAIN ----------------
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
