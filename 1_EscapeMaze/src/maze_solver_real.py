#!/usr/bin/env python3
import math
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# DIFFERENCE WITH THE SIMULATION:
# 1. TURN OFF THE STATE STORAGE MACHANISM (The real condition is much simpler than simulation, so don't need the state storage machanism)
# 2. REVISE THE VELOCITY SPEED AND LINEAR SPEED IN TURNING LEFT SITUATION (Because the maze in reality is much smaller than in simulation)

# BANG BANG CONTROL
class LeftWallFollower:
    def __init__(self):
        """
        Initialize the publisher and subscriber
        Initialize the data points p, and the state of turning left
        """
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb)  

        # Store all lidar data, use scan_cb() to get and refresh data points
        self.p = [9.9] * 360  

        # State of Left Turning: This value is True If and Only If in Turning Left when left side disapear
        self.turn_left_state = False

    def clst_dtc_and_dir(self, start_degree, end_degree):
        """
        Find the closest distance and direction
        """
        min_dtc = self.p[start_degree]
        min_dir = start_degree
        for i in range(start_degree, end_degree):
            if min_dtc > self.p[i]:
                min_dtc = self.p[i]
                min_dir = i
        return min_dtc, min_dir

    def scan_cb(self, msg):
        """
        Scan and get the Lidar data, and store in the list p
        """
        degree = 0
        for i in range(0,360):
            if msg.ranges[degree] == float('inf') or msg.ranges[degree] == 0.0:
                self.p[i] = 9.9  # 9.9 means infinite
            else:
                self.p[i] = msg.ranges[degree]
            degree += 1

    def follow_left_wall(self):
        """
        The Algorithm of Following the left_side: (Bang Bang Control)
        1. If there is no obstacle in front, follow the left side wall, and keep the distance near keep line:
            1-1. If the distance is less than dead distance, move toward keep line (move front and right).
            1-2. If the distance is between the dead line and the bound line, move toward keep line according to the direction of left closest distance.
            1-3. If the distance is larger than bound line, move straight to find a wall.
        2. If there is an obstacle in front:
            2-1. If the left turn state is not true, turn right.
            2-2. If the left turn state is true, turn left.
        """
        twist = Twist()

        left_clst_dtc, left_clst_dir = self.clst_dtc_and_dir(45,135)

        # common use speed
        lvs = 0.2
        avs = 0.2
        av = avs * 3

        # area divided
        dead = 0.2
        keep = 0.3
        bound = 0.5

        # obstacle detect, the range is -15 to 15 degrees
        obstacle_left_detect_dtc, obstacle_left_detect_dir = self.clst_dtc_and_dir(0,15)
        obstacle_right_detect_dtc, obstacle_right_detect_dir = self.clst_dtc_and_dir(345,360)
        obstacle = True if obstacle_left_detect_dtc < keep or obstacle_right_detect_dtc < keep else False
                
        if obstacle:
            print("Obstacle In Front")
            if self.turn_left_state == True:
                twist.linear.x = 0 
                twist.angular.z = av
                self.turn_left_state = True ; print("Hit the obstacle when the state is True, Continue turning left")
            else:
                twist.linear.x = 0
                twist.angular.z = -av
                self.turn_left_state = False ; print("An obstable in front, turn right to avoid it")
        else:
            print("No Obstacle In Front")
            if left_clst_dtc < dead:
                twist.linear.x = lvs 
                twist.angular.z = -avs 
                self.turn_left_state = False ; print("Trying to move along the Keep Line...")
            elif left_clst_dtc < keep:
                if left_clst_dir < 70: # 70 is a special number, means the direction of robot is too toward the wall, so the robot needs turning right to back to keep line
                    twist.linear.x = lvs 
                    twist.angular.z = -avs 
                    self.turn_left_state = False ; print("Trying to move along the Keep Line...")
                elif left_clst_dir > 90: # 90 is the degree of normal left, larger than 90 means the robot is toward away from the wall, so the robot needs turning left to back to keep line
                    twist.linear.x = lvs / 2 ########## REVISED IN REAL SITUATION
                    twist.angular.z = av * 3 ########## REVISED IN REAL SITUATION
                    self.turn_left_state = False ; print("Left Side Disapear, Turn Left")
                else:
                    twist.linear.x = lvs 
                    twist.angular.z = 0
                    self.turn_left_state = False ; print("Trying to move along the Keep Line...")
            elif left_clst_dtc < bound:
                if left_clst_dir < 90: # 90 is the degree of normal left, smaller than 90 means the robot is toward the wall, so the robot nees turning left to back to keep line
                    twist.linear.x = lvs 
                    twist.angular.z = av * 2
                    self.turn_left_state = False ; print("Trying to move along the Keep Line...")
                else:
                    twist.linear.x = lvs / 2 ########## REVISED IN REAL SITUATION
                    twist.angular.z = av * 3 ########## REVISED IN REAL SITUATION
                    self.turn_left_state = False ; print("Left Side Disapear, Turn Left")
            else:
                twist.linear.x = lvs
                twist.angular.z = 0
                self.turn_left_state = False ; print("No obstacle, No left side wall, Keep moving...")
        
        # PUBLISH THE TWIST
        self.cmd_vel_pub.publish(twist)

if __name__ == '__main__':
    rospy.init_node('LeftWallFollower')
    
    LeftWallFollower = LeftWallFollower()

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        LeftWallFollower.follow_left_wall()
        rate.sleep()