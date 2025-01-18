#!/usr/bin/env python
import rospy
import cv2
import cv_bridge  
import numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from obstacle_avoid import LeftWallFollower # this is the last assignment's script, and is introduced and used here

class LineFollowerSim:
    def __init__(self):
        # let image to OpenCV format
        self.bridge = cv_bridge.CvBridge()
        # image callback
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',
                                          Image, self.image_callback)
        # Twist
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

        # the values will update at callback function
        self.cx = -1
        self.cy = -1
        self.middle = -1

        # P control initilization
        self.max_angular_speed = 0.5 # 转的太快了，加一个最大值

        # Update for three chanlenges
        self.obstacle = False
        self.rotation_count = 0
        self.rotation_started = False
        self.explore_state = False

        # Object LeftWallFollower for avoiding the obstacle
        self.left_wall_follower = LeftWallFollower()

    def image_callback(self, msg):
        """
        Callback to `self.image_sub`.
        This method receive the image raw data,
        and process the data to recgnize the yellow line
        """
        # ==============================================
        # PART A: Receive the image from the camera

        # Print message type to verify what is being received (hide, but remain for spare)
        # rospy.loginfo(f"Received image with encoding: {msg.encoding}")
                    
        # Attempt to convert the ROS Image message to an OpenCV image
        rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # OpenCV uses BGR format, so needs change it, if not, the view will recognize yellow as blue
        image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Resize the image
        resized_image = cv2.resize(image_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)

        # show normal image (hide, but keep for spare)
        cv2.imshow('normal_image', resized_image)

        # ================================================
        # PART B: HSV MASK
        # Use HSV images to help recognize yellow line more efficient
        hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        lower_yellow = numpy.array([20, 100, 100]) # Textbook: [50, 50, 170]
        upper_yellow = numpy.array([30, 255, 255]) # Textbook: [255, 255, 190]

        # mask hsv
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        
        # show mask hsv (hide, but keep for spare)
        cv2.imshow('mask_image', mask)

        # masked image with yellow line
        masked_hsv_img = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        
        # show masked image with yellow line (hide, but keep for spare)
        cv2.imshow('masked_hsv_image', masked_hsv_img)

        # =====================================================
        # PART C:  Find the centroid
        h, w, d = resized_image.shape
        search_top = int(3*h/4) # here cause an error, and casting to int avoid that error successfully
        search_bot = search_top + 20
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        # mask image moments
        M = cv2.moments(mask)

        # update cx, cy, w 
        self.cx = -1
        self.cy = -1
        self.middle = w

        # ==============================================
        # detect yellow line and get the centroid circle
        if M['m00'] > 0:
            self.cx = int(M['m10']/M['m00']) # x coordinate of centroid
            self.cy = int(M['m01']/M['m00']) # y coordinate of centroid
            cv2.circle(resized_image,(self.cx,self.cy), 20, (0,0,255), -1) # mark
        else:
            self.cx = -1
            self.cy = -1

        # show circle image
        cv2.imshow('circle', resized_image)

        # this is for cv2.imshow
        cv2.waitKey(3)

    def follow_line(self):
        """
        Follow a yellow line.
        
        Robot starts:
        if yellow line is found:
            Follow
        else if there is no obstacle, no line, explore_state is false (case 1, case 2):
            first rotate 2 circles in place, if there is still no line, enter the state of no line nearby (far away, explore_state = True)
        else if there is no obstacle, no line, explore_state is true (case 1, case 2):
            keep going straight
        else if there is an obstacle
            move along the left wall
        else
            theoretically there is no other situation, but there may be other situations, which are reserved for debugging
        
        Achieve extra functions:
        # 1. Robot starts somewhere where the line is not immediately visible
        # 2. Robot is able to double back along the line allowing it to follow the line infinitely
        # 3. Robot uses Lidar (/scan) to detect an obstacle in the way. Place an object in gazebo of your choice to move around. The more “challenging” choice of obstacle(s) will mean more extra points awarded
        """

        # update obstacle status
        # usage: argument is the detecting distance, over this distance will not detect
        self.obstacle = self.left_wall_follower.is_obstacle_near(0.5)

        if self.cx >= 0 and self.cy >= 0:
            print("SITUATION 1: find yellow line")
            # Reset functional state variables
            self.explore_state = False
            self.rotation_started = False

            # Proportional Control (Only P!)
            err = self.cx - self.middle/2
            self.twist.linear.x = 0.2
            angular_velocity = max(min(-float(err) / 100, self.max_angular_speed), -self.max_angular_speed)
            self.twist.angular.z = angular_velocity 

            print("linear speed: ", self.twist.linear.x)
            print("angular speed:", self.twist.angular.z)

        elif self.cx < 0 and self.cy < 0 and self.obstacle == False and self.explore_state == False:
            # Rotate 720 degrees clockwise in place:
            # When yellow lines appear during the rotation, exit the loop
            print("SITUATION 2: no line, no obstacle, explore_state is False")
            
            if not self.rotation_started:
                # Set rotate value
                self.rotation_started = True
                self.rotation_count = 0

            self.twist.linear.x = 0.0
            self.twist.angular.z = -0.3
            self.rotation_count += 1

            # if find the line, stop rotating
            if self.cx >= 0 and self.cy >= 0:
                print("Yellow line found during rotation, stopping rotation.")
                self.twist.angular.z = 0.0
                self.rotation_started = False
                self.explore_state = False

            # If no line is found after a period of rotation, enter the exploration mode
            elif self.rotation_count > 300:  
                # 300 times is equivalent to about 1.5 rotations
                # rate = 10, 10Hz frequency, angular velocity is 0.3, based on these we can roughly estimate
                print("Rotation completed, entering exploration mode.")
                self.rotation_started = False
                self.explore_state = True

        elif self.cx < 0 and self.cy < 0 and self.obstacle == False and self.explore_state == True:
            print("SITUATION 3: no line, no obstacle, explore_state is True")
            # Exploration mode
            # No rotation, just move forward in a straight line
            self.twist.linear.x = 0.3
            self.twist.angular.z = 0
            if self.cx >= 0 and self.cy >= 0:
                self.explore_state = False

        elif self.cx < 0 and self.cy < 0 and self.obstacle == True:
            print("SITUATION 4: no line, but has obstacle. \nLeft_wall_follower Mode")
            self.twist = self.left_wall_follower.follow_left_wall()

        else:
            # situation should not exist, debug situation
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            print("linear speed: ", 0)
            print("angular speed: ", 0)

        self.cmd_vel_pub.publish(self.twist)

    def run(self):
        """
        Run the Program.
        """
        rate = rospy.Rate(10)
    
        while not rospy.is_shutdown():
            print("\n==================")
            self.follow_line()
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('line_follower_sim')
    follower = LineFollowerSim()
    follower.run()
    
