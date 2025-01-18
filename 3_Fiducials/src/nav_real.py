#!/usr/bin/env python3
import math
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Pose, Twist
from nav_msgs.msg import Odometry
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
        # INITIALIZATION
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # CAMERA PROCESSING
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.image_sub = rospy.Subscriber('/camera/image_processed', Image, self.image_callback)
        self.latest_image = None
        
        # INITIAL POSITION VARIABLES
        self.start_transform = None
        self.is_origin_set = False
        
        # Prepare for TF Tree
        rospy.loginfo("Waiting for TF data...")
        rospy.sleep(1.0)
        
        # Mark the start position
        self.record_start_position()

    def record_start_position(self):
        """
        (TF)
        Use TF to mark the start position
        Not use odom because the fiducial use relative position to camera
        And TF will be more convenient
        """
        try:
            self.start_transform = self.tf_buffer.lookup_transform(
                "odom", "base_footprint", rospy.Time(0), rospy.Duration(5.0)
            )
            self.is_origin_set = True
            rospy.loginfo("Origin position recorded:")
            rospy.loginfo(f"x: {self.start_transform.transform.translation.x:.3f}, "
                         f"y: {self.start_transform.transform.translation.y:.3f}")
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get initial transform: {e}")
            return False

    def image_callback(self, msg):
        """
        Callback function for camera image
        """
        self.latest_image = msg

    def detect_fid(self):
        """
        Detect the fiducial, return its ID.
        
        This function is the key function, 
        and it only return the fiducial occuring in the image. (recently detected fiducial)
        """
        latest_stamp = rospy.Time(0)
        latest_id = False
        
        for fiducial_id in range(4):
            fiducial_frame = f"aruco_marker_{fiducial_id}"
            try:
                transform = self.tf_buffer.lookup_transform(
                    "camera_link", fiducial_frame, rospy.Time(0),
                    rospy.Duration(0.1)
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
        fiducial_frame = f"aruco_marker_{fiducial_id}"
        try:
            transform = self.tf_buffer.lookup_transform(
                "camera_link", fiducial_frame, rospy.Time(0)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            rospy.loginfo(f"Fiducial {fiducial_id} position relative to camera: x={x}, y={y}, z={z}")
            return x, y, z
        except Exception as e:
            rospy.logwarn(f"Could not find fiducial {fiducial_id}: {e}")
            return None

    def get_current_pose(self):
        """
        (Helper Method for moving back to origin)
        Get the robot position relative to odom coordinate
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "base_footprint", rospy.Time(0)
            )
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get current transform: {e}")
            return None

    def get_distance_to_origin(self):
        """
        (Helper Method for moving back to origin)
        Calculate the distance between current position to origin
        """
        if not self.is_origin_set or self.start_transform is None:
            rospy.logwarn("Origin position not set!")
            return None
            
        current_transform = self.get_current_pose()
        if current_transform is None:
            return None
            
        x_diff = (current_transform.transform.translation.x - 
                 self.start_transform.transform.translation.x)
        y_diff = (current_transform.transform.translation.y - 
                 self.start_transform.transform.translation.y)
        return math.sqrt(x_diff ** 2 + y_diff ** 2)

    def get_angle_to_origin(self):
        """
        (Helper Method for moving back to origin)
        Calculate the difference angle for current direction to origin direction
        """
        if not self.is_origin_set or self.start_transform is None:
            rospy.logwarn("Origin position not set!")
            return None
            
        current_transform = self.get_current_pose()
        if current_transform is None:
            return None
            
        x_diff = (self.start_transform.transform.translation.x - 
                 current_transform.transform.translation.x)
        y_diff = (self.start_transform.transform.translation.y - 
                 current_transform.transform.translation.y)
        return math.atan2(y_diff, x_diff)

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
        found_fiducials = set()
        twist = Twist()
        twist.angular.z = 0.5

        while not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            fiducial_id = self.detect_fid()
            if fiducial_id is not False:
                found_fiducials.add(fiducial_id)
                if len(found_fiducials) >= 4:
                    break
            rospy.sleep(0.1)

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
        twist = Twist()
        twist.angular.z = 0.3

        while self.detect_fid() != fiducial_id and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)

        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def face_fiducial(self, fiducial_id):
        """
        Rotates the robot so that it directly faces the detected fiducial.

        Implementation details:
        1. Get the fiducial relative coordinates.
        2. Use P control to rotate the robot to face toward the fiducial.
        3. Stop rotation when x is small enough.

        Eliminate image processing error:
        Use Stable Check, Timeout, Stuck Check parameters and algorithms to eliminate processing error.
        """
        twist = Twist()
        rate = rospy.Rate(10)
        
        # P Control
        Kp = 0.3
        position_threshold = 0.15    
        min_angular_speed = 0.05
        max_angular_speed = 0.3
        
        # Stable Check 
        stable_count = 0
        required_stable_count = 10
        
        # Timeout 
        timeout_count = 0
        max_timeout = 100  # 10s is time over
        
        # Slide average
        x_history = []
        history_size = 3
        
        # Stuck Check
        stuck_count = 0
        max_stuck_count = 20  # 2s no change means stuck
        last_x = None
        
        # Set a maximum attempt chances
        search_direction = None
        search_attempts = 0
        max_search_attempts = 3  
        
        while not rospy.is_shutdown():
            coords = self.get_fid_xyz(fiducial_id)
            if coords is None:
                timeout_count += 1
                # if lost target
                if search_direction is not None:
                    twist.angular.z = 0.2 * search_direction
                    rospy.loginfo(f"Searching direction: {search_direction}")
                else:
                    twist.angular.z = 0.0
                
                if timeout_count > max_timeout:
                    rospy.logwarn(f"Unable to detect fiducial {fiducial_id} for too long")
                    return False
                self.cmd_vel_pub.publish(twist)
                rate.sleep()
                continue

            x, _, z = coords
            rospy.loginfo(f"Raw data - Fiducial {fiducial_id}: x={x:.3f}, z={z:.3f}")
            
            # Stuck check
            # This part is to eliminate the delay error caused by image processing
            if last_x is not None and abs(x - last_x) < 0.01: 
                stuck_count += 1
                if stuck_count > max_stuck_count:
                    rospy.logwarn(f"Robot appears stuck at x={x:.3f}, initiating search pattern")

                    if search_direction is None:
                        search_direction = 1 if x < 0 else -1
                        search_attempts = 0
                    
                    search_attempts += 1
                    if search_attempts > max_search_attempts:
                        search_direction *= -1
                        search_attempts = 0
                        
                    twist.angular.z = 0.2 * search_direction
                    self.cmd_vel_pub.publish(twist)
                    rospy.sleep(0.5)
                    
                    stuck_count = 0
                    x_history.clear()
                    continue
            else:
                stuck_count = 0
                search_direction = None 
            last_x = x
            
            # Update history data
            x_history.append(x)
            if len(x_history) > history_size:
                x_history.pop(0)
                
            x_smooth = sum(x_history) / len(x_history)
            
            rospy.loginfo(f"Face fiducial {fiducial_id} - raw offset: {x:.3f}m, "
                        f"smoothed: {x_smooth:.3f}m, stable_count: {stable_count}, "
                        f"stuck_count: {stuck_count}")

            # Stable Check
            if abs(x_smooth) < position_threshold:
                stable_count += 1
                twist.angular.z = 0.0
                rospy.loginfo(f"Within threshold - stable_count: {stable_count}")
                if stable_count >= required_stable_count:
                    rospy.loginfo(f"Successfully facing fiducial {fiducial_id}")
                    break
            else:
                stable_count = 0
                angular_speed = -Kp * x_smooth
                if abs(angular_speed) < min_angular_speed:
                    angular_speed = min_angular_speed * (-1 if x_smooth < 0 else 1)
                angular_speed = min(max_angular_speed, max(-max_angular_speed, angular_speed))
                twist.angular.z = angular_speed
                
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Finally Stop
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(1.0)
        
        # Finally Check
        final_coords = self.get_fid_xyz(fiducial_id)
        if final_coords is not None:
            x, _, z = final_coords
            rospy.loginfo(f"Final position - Fiducial {fiducial_id}: x={x:.3f}, z={z:.3f}")
            if abs(x) > position_threshold:
                rospy.logwarn(f"Warning: Final position offset ({x:.3f}) > threshold ({position_threshold})")
        
        return True
                
    def move_to_fiducial(self, fiducial_id):
        """
        Move toward the fiducial, stop in front of the fiducial.

        Implementation: Use different speed when getting closer, faster when it is far away

        Eliminate image processing delay: Use lost detect and low speed counter to eliminate delay error.
        """
        twist = Twist()
        rate = rospy.Rate(10)
        Kp_angular = 0.3
        
        # Speed and distance parameter
        fast_speed = 0.3      
        slow_speed = 0.1      
        stop_distance = 0.3   
        slow_distance = 0.4   
    
        # Low speed counter
        slow_move_start = None
        slow_move_duration = rospy.Duration(3.0)  # maximum for low speed moving
    
        # Target Lost Detection
        last_coords = None
        lost_count = 0
        max_lost_count = 10  
        same_data_count = 0
        max_same_data_count = 5 
        
        while not rospy.is_shutdown():
            coords = self.get_fid_xyz(fiducial_id)
            
            # Detect if the target is lost
            if coords is None:
                lost_count += 1
                if lost_count > max_lost_count:
                    rospy.logwarn(f"Lost fiducial {fiducial_id} for too long (1s), stopping")
                    break
                twist.angular.z = 0.0
                twist.linear.x = 0.0
                self.cmd_vel_pub.publish(twist)
                rate.sleep()
                continue

            x, _, z = coords
            
            # Check the data update
            if last_coords is not None:
                if (abs(x - last_coords[0]) < 0.0001 and 
                    abs(z - last_coords[2]) < 0.0001):  
                    same_data_count += 1
                    if same_data_count > max_same_data_count:
                        rospy.logwarn("No new data for 0.5s, considering target lost")
                        break
                else:
                    same_data_count = 0
                    lost_count = 0  
            last_coords = (x, 0, z)

            rospy.loginfo(f"Move to fiducial {fiducial_id} - distance: {z:.3f}m, offset: {x:.3f}m")
            
            # Check if is close enough
            if z <= stop_distance:
                rospy.loginfo(f"Reached target distance: {z:.3f}m")
                break
                
            # If move too far away from original target...
            if abs(x) > 0.15:
                rospy.logwarn(f"Large offset detected ({x:.3f}m), readjusting...")
                self.face_fiducial(fiducial_id)
                continue

            # Angular Speed
            twist.angular.z = -Kp_angular * x
            
            # Use different speed in different distances
            if z > slow_distance:
                twist.linear.x = fast_speed
                slow_move_start = None
            else:
                twist.linear.x = slow_speed
                if slow_move_start is None:
                    slow_move_start = rospy.Time.now()
                elif (rospy.Time.now() - slow_move_start) > slow_move_duration:
                    rospy.loginfo("Slow movement timeout reached")
                    break
            
            # Set minimum angular speed to avoid not rotating
            twist.angular.z = min(0.5, max(-0.5, twist.angular.z))
            
            rospy.loginfo(f"Speed: {twist.linear.x:.3f} m/s, Angular: {twist.angular.z:.3f} rad/s")
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Make sure it is fully stopped
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        # Final position confirm
        final_coords = self.get_fid_xyz(fiducial_id)
        if final_coords is not None:
            _, _, final_z = final_coords
            rospy.loginfo(f"Final distance to fiducial: {final_z:.3f}m")
        
        return True

    def back_to_origin(self):
        """
        Move back to the origin
        """
        if not self.is_origin_set:
            rospy.logerr("Cannot return to origin: Origin position not set!")
            return False

        twist = Twist()
        rate = rospy.Rate(10)
        Kp_linear = 0.3
        Kp_angular = 0.5
        
        distance_threshold = 0.05
        angle_threshold = 0.1
        
        # Innter this distance, cut speed enormous
        slow_distance = 0.1  
        
        # Low speed detect parameter
        min_speed_threshold = 0.05  
        low_speed_start = None
        low_speed_duration = rospy.Duration(1.0)  
        
        start_time = rospy.Time.now()
        timeout_duration = rospy.Duration(30)

        while not rospy.is_shutdown():
            # Timeover Check
            if (rospy.Time.now() - start_time) > timeout_duration:
                rospy.logwarn("Back to origin timeout reached")
                break

            # Get current position
            current_transform = self.get_current_pose()
            if current_transform is None:
                continue

            # Calculate distance and angular
            distance = self.get_distance_to_origin()
            if distance is None:
                continue

            # Get current direction
            current_quat = current_transform.transform.rotation
            _, _, yaw = euler_from_quaternion([
                current_quat.x, current_quat.y, current_quat.z, current_quat.w
            ])

            # Calculate target angular
            target_angle = self.get_angle_to_origin()
            if target_angle is None:
                continue

            # Calculate angle error
            angle_error = target_angle - yaw
            if angle_error > math.pi:
                angle_error -= 2 * math.pi
            elif angle_error < -math.pi:
                angle_error += 2 * math.pi

            rospy.loginfo(f"Distance to origin: {distance:.3f}m, "
                        f"Angle error: {math.degrees(angle_error):.1f}°")

            # Check distance threshold
            if distance < distance_threshold:
                rospy.loginfo("Reached origin point")
                break

            # Control moving
            twist.angular.z = Kp_angular * angle_error
            
            # Control angular speed
            if abs(angle_error) < angle_threshold:
                target_linear_speed = Kp_linear * distance
                # Slow down when near to the target
                if distance < slow_distance:
                    target_linear_speed *= (distance / slow_distance)
                
                twist.linear.x = min(0.3, max(0.0, target_linear_speed))
                
                # Check if is slow mode
                if twist.linear.x < min_speed_threshold:
                    if low_speed_start is None:
                        low_speed_start = rospy.Time.now()
                        rospy.loginfo("Entered low speed mode")
                    elif (rospy.Time.now() - low_speed_start) > low_speed_duration:
                        rospy.loginfo("Low speed maintained for 1s, stopping")
                        break
                else:
                    low_speed_start = None
            else:
                twist.linear.x = 0.0
                low_speed_start = None  # Reset

            # Set minimum angular speed
            twist.angular.z = min(0.5, max(-0.5, twist.angular.z))

            rospy.loginfo(f"Linear speed: {twist.linear.x:.3f} m/s")
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop moving
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)

        print("=====================")
        print("BACK TO BASE POINT")
        return True

if __name__ == '__main__':
    try:
        rospy.init_node('nav_real')
        nav = NavReal()
        if nav.is_origin_set:
            nav.scan_for_fids()
            target_ids = [0, 1, 2, 3]
            for id in target_ids:
                print("NOW START MOVING TO: FIDUCIAL-", id)
                nav.match_fiducial_rotation(id)
                nav.face_fiducial(id)
                nav.move_to_fiducial(id)
                nav.back_to_origin()
        else:
            rospy.logerr("Failed to initialize NavReal: Could not record origin position")
    except rospy.ROSException as e:
        rospy.logerr(f"Error in NavReal: {e}")