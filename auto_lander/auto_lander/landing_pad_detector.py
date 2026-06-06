#!/usr/bin/env python3
import os
import sys
import signal
from datetime import datetime

import cv2
import numpy as np
from cv_bridge import CvBridge
import tf2_ros
import tf_transformations
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
import rclpy.duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from mavros_msgs.srv import CommandLong


@dataclass
class TagDefinition:
    """Defines the ArUCo tag definition."""

    size: float
    position: tuple[float, float, float]
    object_points: np.ndarray = field(init=False)

    def __post_init__(self):
        half = self.size / 2.0

        self.object_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)


class VisionPerception(Node):
    """ Defines the vision perception node"""
    def __init__(self):
        """ Initialise the vision perception node
        """
        super().__init__("landing_pad_detection_node")

        # ---- PARAMETERS ----
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

        # ---- CAMERA PARAMETERS ----
        # Must match the IR input size you exported (default IRIS is 640 x 480)
        self.declare_parameter("imgsz_width", 640)
        self._image_width = int(self.get_parameter("imgsz_width").get_parameter_value().integer_value)
        
        self.declare_parameter("imgsz_height", 480)
        self._image_height = int(self.get_parameter("imgsz_height").get_parameter_value().integer_value)

        self._camera_fov_horizontal = 2.0  # radians (≈114.6°) – tune for your camera
        self._camera_fov_vertical = 2 * np.arctan(np.tan(self._camera_fov_horizontal / 2) / (self._image_width/self._image_height))

        # Generate the camera matrix
        fx = self._image_width / (2 * np.tan(self._camera_fov_horizontal / 2))
        fy = self._image_height / (2 * np.tan(self._camera_fov_vertical / 2))
        self._camera_matrix = np.array([
            [fx, 0, self._image_width/2],
            [0, fy, self._image_height/2 ],
            [0, 0, 1],
        ], dtype=np.float64)

        self._dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)

        # ---- GIMBAL CONTROLLER PARAMETERS ----
        self._gimbal_Kp = 0.01
        self._servo_angle = -90.0

        self._gimbal_servo_ID = 10

        self._servo_min_angle = -135.0
        self._servo_max_angle = 45.0

        self._servo_pwm_min = 1100
        self._servo_pwm_max = 1900

        self._servo_response_delay = 0.025

        self._image_timer_rate = 1.0/30 # 30 FPS

        # ---- TAG PARAMETERS ----
        self._tags = {
            35: TagDefinition(
                size=0.541,
                position=(0.0, 0.3700, 0.0)
            ),
            27: TagDefinition(
                size=0.081,
                position=(0.0, 0.0000, 0.0)
            ),
            0: TagDefinition(
                size=0.081,
                position=(0.0, 0.7400, 0.0)
            ),
        }
        # ---- SUBSCRIPTIONS ----
        _img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._webcam_publisher = self.create_publisher(Image, "/image", 10)
        self._landing_pad_found_publisher = self.create_publisher(Bool, "/landing_pad/found", 10)

        # ---- SERVICES ----
        self.client = self.create_client(CommandLong, '/mavros/cmd/command')

        # ---- TF2 ----
        self._tf_quad_to_cam_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._tf_cam_to_tag_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._tf_tag_to_landing_pad_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- OPENCV ----
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_params.adaptiveThreshWinSizeMin = 3
        self._aruco_params.adaptiveThreshWinSizeMax = 250  # default is 23 — increase this significantly
        self._aruco_params.adaptiveThreshWinSizeStep = 10
        self._aruco_params.minMarkerPerimeterRate = 0.01   # default 0.03 — allow smaller apparent perimeter
        self._aruco_params.maxMarkerPerimeterRate = 4.0    # default 4.0 — already fine
        self._aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        # ---- INITIALISATION ----
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
                Image, "/camera/image_raw", self._image_callback, _img_qos
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
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._image_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._image_height)
                
                # Log actual camera properties
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                self.get_logger().info(
                    f"ArUCoImageNode started in WEBCAM mode. Camera properties: "
                    f"FPS={actual_fps}, Width={actual_width}, Height={actual_height}"
                )
            self.timer = self.create_timer(self._image_timer_rate, self._webcam_timer_callback)  # 30 Hz


    # ---- GIMBAL CONTROLLER IMPLEMENTATIONS ----
    def _send_servo_command(self, servo_id, pwm_value):
        """ Sends servo command via Mavlink

        :param servo_id: Servo ID to actuate
        :param pwm_value: PWM value to command to servo
        """
        req = CommandLong.Request()
        req.command = 183            # MAV_CMD_DO_SET_SERVO
        req.param1 = float(servo_id) # Servo channel number (1 to 16)
        req.param2 = float(pwm_value)# PWM value (typically 1000 to 2000)

        self.client.call_async(req)


    def _gimbal_controller(self, image_points):
        """ Determines gimbal required output to centre on tag

        :param image_points: Tag corner points
        """
        # Controller to centre the landing pad into the vertical centre of the image
        centre_y = np.mean(image_points[:, 1])
        image_centre_y = self._image_height / 2
        tag_error = centre_y - image_centre_y   # error from centre of image

        # P-Controller
        self._servo_angle = self._servo_angle - self._gimbal_Kp * tag_error
        self._servo_angle = self._servo_angle = np.clip(self._servo_angle, self._servo_min_angle, self._servo_max_angle) # limits


    def _gimbal_publisher(self, servo_angle, stamp):
        """ Publish the commanded gimbal angle to the gimbal as a PWM signal

        :param servo_angle: Desired gimbal angle to servo
        """
        # Map to PWM
        pwm = int(((servo_angle - self._servo_min_angle) / 180.0) * (self._servo_pwm_max - self._servo_pwm_min) + self._servo_pwm_min)
        pwm = np.clip(pwm, self._servo_pwm_min, self._servo_pwm_max)

        self._send_servo_command(self._gimbal_servo_ID, pwm)

        # Publish the gimbal servo angle to the quad -> cam transformation
        quad_to_cam_tf_msg = self.quad_to_cam_transformstamped(stamp, servo_angle)

        self._tf_quad_to_cam_broadcaster.sendTransform(quad_to_cam_tf_msg)


    # ---- IMAGE CALLBACK IMPLEMENTATIONS ----
    def _webcam_timer_callback(self):
        """ Read image from webcam and publish to /image topic
        """
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to IR input size (square) – must match export
                frame = cv2.resize(
                    frame, (self._image_width, self._image_height), interpolation=cv2.INTER_NEAREST
                )
                msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()

                self._image_callback(msg)
            else:
                self.get_logger().warning("Failed to read frame from webcam.")
        else:
            self.get_logger().warning("Webcam not opened.")


    def _image_callback(self, msg):
        """ Process image and detect tag, calculate pose and publish tf_transform.

        :param msg: Frame from camera
        """
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = msg.header.stamp

        # Ensure inference size matches IR (handles topic frames of any size)
        if frame.shape[0] != self._image_height or frame.shape[1] != self._image_width:
            frame = cv2.resize(frame, (self._image_width, self._image_height), interpolation=cv2.INTER_LINEAR)

        # Inference (ArUCo detection via OpenCV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Change to greyscale before inference step
        corners, ids, _ = self.detector.detectMarkers(gray_frame)

        # Check that the tag_id is recognised for tag_id_arr in ids:
        valid_indices = []
        if ids is not None:
            valid_indices = [i for i, tag_id_arr in enumerate(ids) if int(tag_id_arr[0]) in self._tags]

        landing_pad_found = len(valid_indices) > 0
        self._landing_pad_found_publisher.publish(Bool(data=landing_pad_found))

        # If tag is recognised, then execute pose calculations
        if landing_pad_found:
            # Compute areas
            idx = max(valid_indices, key=lambda i: cv2.contourArea(corners[i][0].astype(np.float32)))
            tag_id = int(ids[idx][0])

            image_points = corners[idx][0].astype(np.float32)
            tag = self._tags[tag_id]
            object_points = tag.object_points

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
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
                        tag.size * 0.5
                    )

                # Broadcast landing_pad position relative to camera frame
                cam_to_tag_tf_msg = self.cam_to_tag_transformstamped(stamp, tag_id, rvec, tvec)
                tag_to_landing_pad_tf_msg = (self.tag_to_landing_pad_transformstamped(stamp, tag_id, tag.position))
                if cam_to_tag_tf_msg is not None:
                    self._tf_cam_to_tag_broadcaster.sendTransform(cam_to_tag_tf_msg)
                    self._tf_tag_to_landing_pad_broadcaster.sendTransform(tag_to_landing_pad_tf_msg)

        # Publish the gimbal angle regardless of if the pose update was good
        self._gimbal_publisher(self._servo_angle, stamp)
        
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
            msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self._webcam_publisher.publish(msg)

    # ---- HELPER FUNCTIONS ----
    def create_video_from_frames(self):
        """ Create video from saved frames
        """
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
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
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

    @staticmethod
    def quad_to_cam_transformstamped(stamp, servo_angle):
        """ Generate the quad to cam TF transfrom

        :param servo_angle: Servo angle to pass into TF
        """
        # From gimbal angle, find the quad --> cam transform (4x4)
        t_quad_to_cam_pad = [0.0, 0.0, -0.1249]
        q_tag_to_landing_pad = tf_transformations.quaternion_from_euler(
            -1.5707963 + np.deg2rad(servo_angle), 0.0, -1.5707963
        )

        # Header for pose
        tf_quad_to_cam = TransformStamped()
        tf_quad_to_cam.header.stamp = stamp
        tf_quad_to_cam.header.frame_id = "base_link"
        tf_quad_to_cam.child_frame_id = "camera_link"
        tf_quad_to_cam.transform.translation.x = t_quad_to_cam_pad[0]
        tf_quad_to_cam.transform.translation.y = t_quad_to_cam_pad[1]
        tf_quad_to_cam.transform.translation.z = t_quad_to_cam_pad[2]
        tf_quad_to_cam.transform.rotation.x = q_tag_to_landing_pad[0]
        tf_quad_to_cam.transform.rotation.y = q_tag_to_landing_pad[1]
        tf_quad_to_cam.transform.rotation.z = q_tag_to_landing_pad[2]
        tf_quad_to_cam.transform.rotation.w = q_tag_to_landing_pad[3]

        return tf_quad_to_cam
    

    @staticmethod
    def cam_to_tag_transformstamped(stamp, tag_id, rvec, tvec):
        """ Generate the cam to tag TF transfrom

        :param tag_id: Tag ID
        :param rvec: Tag rotation
        :param tvec: Tag translation
        """
        # From ArUCo tag, find the camera --> tag transform (4x4)
        t_cam_to_tag = tvec.reshape(3) # position
        R_cam_to_tag, _ = cv2.Rodrigues(rvec)
        T_cam_to_tag = np.eye(4)
        T_cam_to_tag[:3, :3] = R_cam_to_tag
        q_cam_to_tag = tf_transformations.quaternion_from_matrix(T_cam_to_tag)

        # Header for pose
        tf_cam_to_tag = TransformStamped()
        tf_cam_to_tag.header.stamp = stamp
        tf_cam_to_tag.header.frame_id = "camera_link"
        tf_cam_to_tag.child_frame_id = f"tag{tag_id}_link" # keeps tag_id positions in sync with cam detection
        tf_cam_to_tag.transform.translation.x = t_cam_to_tag[0]
        tf_cam_to_tag.transform.translation.y = t_cam_to_tag[1]
        tf_cam_to_tag.transform.translation.z = t_cam_to_tag[2]
        tf_cam_to_tag.transform.rotation.x = q_cam_to_tag[0]
        tf_cam_to_tag.transform.rotation.y = q_cam_to_tag[1]
        tf_cam_to_tag.transform.rotation.z = q_cam_to_tag[2]
        tf_cam_to_tag.transform.rotation.w = q_cam_to_tag[3]

        return tf_cam_to_tag
    

    @staticmethod
    def tag_to_landing_pad_transformstamped(stamp, tag_id, tag_position):
        """ Generate the tag to landing pad TF transfrom

        :tag_id: Tag ID
        :tag_positions: Tag position on landing pad
        """
        # Broadcast the tag_to_landing_pad tf transform
        t_tag_to_landing_pad = np.array(tag_position)
        q_tag_to_landing_pad = tf_transformations.quaternion_from_euler(0.0, 0.0, -1.570796326) # Turns out all the tags were 90deg off...

        # Header for pose
        tf_tag_to_landing_pad = TransformStamped()
        tf_tag_to_landing_pad.header.stamp = stamp
        tf_tag_to_landing_pad.header.frame_id = f"tag{tag_id}_link" # keeps tag_id positions in sync with cam detection
        tf_tag_to_landing_pad.child_frame_id = "landing_pad_link"
        tf_tag_to_landing_pad.transform.translation.x = t_tag_to_landing_pad[0]
        tf_tag_to_landing_pad.transform.translation.y = t_tag_to_landing_pad[1]
        tf_tag_to_landing_pad.transform.translation.z = t_tag_to_landing_pad[2]
        tf_tag_to_landing_pad.transform.rotation.x = q_tag_to_landing_pad[0]
        tf_tag_to_landing_pad.transform.rotation.y = q_tag_to_landing_pad[1]
        tf_tag_to_landing_pad.transform.rotation.z = q_tag_to_landing_pad[2]
        tf_tag_to_landing_pad.transform.rotation.w = q_tag_to_landing_pad[3]

        return tf_tag_to_landing_pad

def get_workspace_root():
    """ Find the workspace root by looking for colcon workspace structure
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'src')) and \
           os.path.exists(os.path.join(current_dir, 'build')) and \
           os.path.exists(os.path.join(current_dir, 'install')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = VisionPerception()
    
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
