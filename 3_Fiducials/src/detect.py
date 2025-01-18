#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import tf2_ros
import tf 
from tf import transformations 
import geometry_msgs.msg

class ArucoDetector:
    """
    This class is the substitute of package 'aruco_detect'

    Note: This script is finished with ChatGPT and Claude, otherwise it will be a too complicated job for me.

    Why I don't use 'aruco_detect' package?
    ----Because my turtlebot3 uses an external USB Camera, which is not compatible with ROS
        So I wrote an image processing script directly on the Raspberry Pi of turtlebot3
        And published the image on '/camera/image_processed'
        The aruco_detect package that comes with ROS is not very useful
        So I made a fiducial_detect script myself

    The function of this script:
    1. Subscribes to the '/camera/image_processed' topic to receive processed images from the external USB camera.
    2. Detects ArUco markers in the images using OpenCV's ArUco module.
    3. Estimates the pose (position and orientation) of detected markers relative to the camera.
    4. Broadcasts the pose of each detected marker as a TF transform.
    5. Publishes the processed image with marker annotations on the 'aruco_detected_image' topic.

    Attributes:
        aruco_dict: The dictionary of ArUco markers used for detection.
        aruco_params: Parameters for the ArUco marker detection algorithm.
        marker_size: The size of the ArUco marker in meters.
        camera_matrix: The intrinsic parameters of the camera.
        dist_coeffs: The distortion coefficients of the camera.
        tf_broadcaster: A TF broadcaster to publish the marker's position and orientation.
        image_pub: A ROS publisher for the processed image with marker annotations.
        image_sub: A ROS subscriber to receive images from the '/camera/image_processed' topic.

    Usage:
        This script should be run as a ROS node. It listens for images published on '/camera/image_processed',
        detects ArUco markers, and publishes their pose as TF transforms while providing visual feedback via
        annotated images.
    """

    def __init__(self):
        rospy.init_node('aruco_detector')
        self.bridge = CvBridge()
        
        # ArUco Settings
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 0.14  # 14cm
        
        # Camera Parameters
        self.camera_matrix = np.array([[570.0, 0, 320.0],
                                     [0, 570.0, 240.0],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        
        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Publisher
        self.image_pub = rospy.Publisher('aruco_detected_image', Image, queue_size=10)
        
        # Sub Camera Image, which is processed on Raspberry Pi with OpenCV
        self.image_sub = rospy.Subscriber('/camera/image_processed', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(
                cv_image, 
                self.aruco_dict,
                parameters=self.aruco_params
            )
            
            if ids is not None:
                # draw the marker detected
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # estimate the pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    self.marker_size,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                for i in range(len(ids)):
                    # draw axis
                    cv2.aruco.drawAxis(
                        cv_image,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvecs[i],
                        tvecs[i],
                        0.1
                    )
                    
                    # publish TF
                    t = geometry_msgs.msg.TransformStamped()
                    t.header.stamp = rospy.Time.now()
                    t.header.frame_id = "camera_link"
                    t.child_frame_id = f"aruco_marker_{ids[i][0]}"
                    
                    # set positions
                    t.transform.translation.x = tvecs[i][0][0]
                    t.transform.translation.y = tvecs[i][0][1]
                    t.transform.translation.z = tvecs[i][0][2]
                    
                    # Convert from rotation vector to quaternion
                    rot_matrix, _ = cv2.Rodrigues(rvecs[i])
                    rot_matrix_4x4 = np.eye(4)
                    rot_matrix_4x4[:3, :3] = rot_matrix
                    quat = transformations.quaternion_from_matrix(rot_matrix_4x4)  # 修改这行
                    
                    t.transform.rotation.x = quat[0]
                    t.transform.rotation.y = quat[1]
                    t.transform.rotation.z = quat[2]
                    t.transform.rotation.w = quat[3]
                    
                    # publish TF
                    self.tf_broadcaster.sendTransform(t)
                    
                    # position
                    x, y, z = tvecs[i][0]
                    rospy.loginfo(f"Marker {ids[i][0]} position (meters):")
                    rospy.loginfo(f"X: {x:.3f} Y: {y:.3f} Z: {z:.3f}")
                    
                    # show information (coordinates) on image
                    cv2.putText(cv_image,
                              f"ID:{ids[i][0]} x:{x:.2f} y:{y:.2f} z:{z:.2f}m",
                              (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 255, 0),
                              2)
            
            # Publish the image with fiducial detection marks
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        detector = ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass