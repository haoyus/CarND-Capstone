#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint',Int32,self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint',Int32,self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.waypoints_2D = None
        self.waypoints_tree = None
        self.pose = None

        self.stopline_wp_idx = -1

        self.loop()

        #print("Waypoint Updater started!")
        rospy.spin()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                #closest_waypoint_index = self.get_closest_waypoint_index()
                self.publish_waypoints()
                print("Waypoint Updater publishing!")
            rate.sleep()

    def get_closest_waypoint_index(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_index = self.waypoints_tree.query([x,y],1)[1]
        closest_coord = self.waypoints_2D[closest_index]
        prev_coord = self.waypoints_2D[(closest_index-1)%len(self.waypoints_2D)]

        cl_vec = np.array(closest_coord)
        pr_vec = np.array(prev_coord)
        pos_vec = np.array([x,y])

        if np.dot(cl_vec-pr_vec,pos_vec-cl_vec)>0:
            closest_index = (closest_index+1)%len(self.waypoints_2D)
        
        return closest_index

    def publish_waypoints(self):
        # wps = Lane()
        # wps.header = self.base_waypoints.header
        # wps.waypoints = self.base_waypoints.waypoints[cl_index : cl_index+LOOKAHEAD_WPS]
        # self.final_waypoints_pub.publish(wps)

        final_path = self.generate_path()
        self.final_waypoints_pub.publish(final_path)


    def generate_path(self):
        path = Lane()
        closest_wp_idx = self.get_closest_waypoint_index()
        farthest_wp_idx = closest_wp_idx + LOOKAHEAD_WPS
        base_wps = self.base_waypoints.waypoints[closest_wp_idx:farthest_wp_idx]
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx>=farthest_wp_idx):
            path.waypoints = base_wps
        else:
            path.waypoints = self.decelerate(base_wps,closest_wp_idx)

        return path

    def decelerate(self,wps,closest_idx):
        tmp = []
        for i,wp in enumerate(wps):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx -3,0)
            dist = self.distance(wps,i,stop_idx)
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel<1.0:
                vel = 0
            p.twist.twist.linear.x = min(vel,wp.twist.twist.linear.x)
            
            tmp.append(p)
        
        return tmp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if None==self.waypoints_2D:
            self.waypoints_2D = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2D)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data
        #print("Waypoint_Updater: got traffic wpt and set self stopline_wp_idx ",msg)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
