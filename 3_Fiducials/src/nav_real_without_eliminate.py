#!/usr/bin/env python3

import math
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Pose, Twist
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

class NavReal:
    """
    This class control a robot to detect fiducials and move in front of them one by one.

    This class allows the robot achieve the goal by following sequence of tasks:
    1. Find all four fiducials which has unique ID. 
    2. Move to each unique fiducial one by one, with the following instructions:
        (1) Rotate until find the fiducial with smallest ID.
        (2) Face toward the fiducial.
        (3) Move toward the fiducial, and stop in front of that fiducial.
        (4) Move back to the origin point, continue to find next fiducial with larger ID.
    3. After moving in front of all four fiducials, finish the task.

    Key Methods:
        detect_fid():
            Detects the most recently observed fiducial and returns its ID. 

        get_fid_xyz(fiducial_id):
            Return the (x, y, z) position of a fiducial relative to the camera. 

        scan_for_fids():
            Rotates the robot to scan to make sure all fiducials are in the environment.

        match_fiducial_rotation(fiducial_id):
            Rotates the robot until it aligns its orientation with the fiducial with certain ID.

        face_fiducial(fiducial_id):
            Allow the robot face toward the fiducial with certain ID.

        move_to_fiducial(fiducial_id):
            Moves the robot in front of the fiducial with certain ID.

        back_to_origin():
            Let the robot back to origin using odometry.
    """

    def __init__(self):
        self.my_odom_sub = rospy.Subscriber('my_odom', Point, self.my_odom_cb)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Add camera-related initialization
        # Add this to help detect_fid function work properly, see detect_fid function intro.
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.image_sub = rospy.Subscriber('/camera/image_processed', Image, self.image_callback)
        self.latest_image = None

    def image_callback(self, msg):
        self.latest_image = msg
       
    def my_odom_cb(self, msg):
        """
        Callback for `my_odom` topic. Updates the robot's distance and yaw.
        """
        self.current_distance = msg.x  # Euclidean distance the robot moves
        self.current_yaw = msg.y       # Current heading angle (radians)

    def detect_fid(self):
        """
        Detect the fiducial, return its ID.
        
        This function is the key function, 
        and it only return the fiducial occuring in the image. (recently detected fiducial)
        """
        # Since the identified fiducial will be published on tf, 
        # and there may be more than one fiducial coordinate at the same time, 
        # the most recently appeared fiducial is returned as the only identified fiducial.
        latest_stamp = rospy.Time(0)
        latest_id = False
        
        # Here '4' is the amount of fiducials and its ID in this case
        # Beacuse I use fiducials with ID = 0, 1, 2, 3
        # If the fiducials have larger or different ID
        # This range can be modified, or set a dynamic variable
        for fiducial_id in range(4):
            fiducial_frame = f"aruco_marker_{fiducial_id}"
            try:
                transform = self.tf_buffer.lookup_transform(
                    "camera_link", fiducial_frame, rospy.Time(0),
                    rospy.Duration(0.1)  # avoid blocking
                )
                if transform.header.stamp > latest_stamp:
                    latest_stamp = transform.header.stamp
                    latest_id = fiducial_id
            except Exception:
                continue
                
        return latest_id

    
    def get_fid_xyz(self, fiducial_id):
        """
        Get fiducial (e.g., the hand) coordinate information (x, y, z) relative to the camera.
        This coordinate is relative to the camera, and:
            z is the distance between the fiducial and the robot
            x is the distance between the ceter of the camera with the fiducial image detected.

        This function helps the robot move toward the fiducial,
        it is pretty similar to line following, 
        let robot follow the center of the fiducial mark with P control.
        """
        fiducial_frame = f"aruco_marker_{fiducial_id}"  # 构造 fiducial 的 TF frame 名称

        try:
            # 查询 TF Tree 中 fiducial 相对于相机的变换
            transform = self.tf_buffer.lookup_transform(
                "camera_link", fiducial_frame, rospy.Time(0)
            )

            # 提取平移信息（xyz）
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            rospy.loginfo(f"Fiducial {fiducial_id} position relative to camera: x={x}, y={y}, z={z}")
            return x, y, z

        except Exception as e:
            rospy.logwarn(f"Could not find fiducial {fiducial_id}: {e}")
            return None
        
        
    def scan_for_fids(self):
        """
        Scans for fiducials by rotating in place. When a fiducial is detected,
        logs its ID and odom coordinates. Stops after finding 4 unique fiducials.

        Implementation details:
        1. Let the robot rotate at origin.
        2. Use a set to remember the fiducials recognized.
        3. If the length of the set is larger than 4, 
           which means all fiducials with unique ID are recognized, stopped.
        """
        # Use a set to remember all unique fiducials (with unique ID)
        found_fiducials = set()

        # Let the robot rotate at origin.
        twist = Twist()
        twist.angular.z = 0.5

        # Use while loop to iterate, and out the loop when all four fiducials are found.
        while True:
            self.cmd_vel_pub.publish(twist)
            fiducial_id = self.detect_fid()
            if fiducial_id is not False:
                found_fiducials.add(fiducial_id) 
                if len(found_fiducials) >= 4:
                    break
            rospy.sleep(0.1)

        # Stop moving
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        
        print("FINISH SCANNING, FOUR FIDUCIALS FOUND")

    
    def match_fiducial_rotation(self, fiducial_id):
        """
        To find a fiducial with certain ID (parameter fiducial_id)

        Implementation details:
        1. Keep rotating if not find the fiducial with certain ID
        2. Stop rotationg if found.        
        """

        # Rotates the robot so that its `base_link` frame's orientation matches
        # that of the detected fiducial's frame.
        twist = Twist()
        twist.angular.z = 0.3

        while self.detect_fid() != fiducial_id:
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)

        # Stop rotation
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)


    def face_fiducial(self, fiducial_id):
        """
        Rotates the robot so that it directly faces the detected fiducial.

        Implementation details:
        1. Get the fiducial relative coordinates.
        2. Use P control to rotate the robot to face toward the fiducial.
        3. Stop rotation when x is small enough.
        """
        twist = Twist()
        rate = rospy.Rate(10)  # 10Hz control loop
        Kp = 0.5  # Proportional control gain

        while not rospy.is_shutdown():
            # use get_fid_xyz to get x,y,z 
            coords = self.get_fid_xyz(fiducial_id)
            if coords is None:
                continue

            # if x is not smaller than 0.2, use p control to rotate 
            x, _, _ = coords
            if abs(x) < 0.2:
                break

            twist.angular.z = -Kp * x  # Negative because positive x means turn left
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop rotation when x is small enough
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)


    def move_to_fiducial(self, fiducial_id):
        """
        Moves the robot towards the detected fiducial.

        Precondition: the robot is face the fiducial with certain ID.

        Implementation details:
        1. Use P control to move toward the fiducial
        2. If the robot is close enough to the fiducial, stop moving.
        """
        twist = Twist()
        rate = rospy.Rate(10)
        Kp_angular = 0.5  # Angular velocity control gain
        Kp_linear = 0.3   # Linear velocity control gain

        while not rospy.is_shutdown():
            coords = self.get_fid_xyz(fiducial_id)
            if coords is None:
                continue

            # The distance that can be recognized as close enough
            x, _, z = coords
            if z <= 0.4:
                break

            # P control for both rotation and forward motion
            twist.angular.z = -Kp_angular * x
            twist.linear.x = Kp_linear * (z - 0.2)  # Slow down as we approach target
            # twist.linear.x = 0.3
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop all motion
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)


    def back_to_origin(self):
        """
        Move the robot back to the origin using the odom frame.

        Function implementation details:
        1. Get current position relative to odom
        2. Calculate angle to origin and rotate
        3. Move to origin
        """
        twist = Twist()
        rate = rospy.Rate(10)
        Kp_linear = 0.3
        Kp_angular = 0.5

        while not rospy.is_shutdown():
            try:
                # 1. Get current position relative to odom
                transform = self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time(0))
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                
                # 2. Calculate angle to origin and rotate
                angle_to_origin = math.atan2(-y, -x)
                quaternion = [transform.transform.rotation.x, transform.transform.rotation.y,
                                transform.transform.rotation.z, transform.transform.rotation.w]
                _, _, yaw = euler_from_quaternion(quaternion)
                
                angle_error = angle_to_origin - yaw
                distance = math.sqrt(x*x + y*y)

                if distance < 0.1:  # At origin
                    break

                # 3. Move to origin
                twist.angular.z = Kp_angular * angle_error
                if abs(angle_error) < 0.3:  # Only move when facing right direction
                    twist.linear.x = Kp_linear * distance
                else:
                    twist.linear.x = 0.0

                self.cmd_vel_pub.publish(twist)
                rate.sleep()

            except Exception as e:
                continue

        # move to origin, stop moving
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)

        print("=====================")
        print("BACK TO BASE POINT")



if __name__ == '__main__':
    rospy.init_node('nav_real')

    nav = NavReal()
    nav.scan_for_fids()
    target_ids = [0, 1, 2, 3]
    
    for id in target_ids:
        print("NOW START MOVING TO: FIDUCIAL-", id)
        nav.match_fiducial_rotation(id)
        nav.face_fiducial(id)
        nav.move_to_fiducial(id)
        nav.back_to_origin()
    