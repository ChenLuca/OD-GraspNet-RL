#!/usr/bin/env python3
import sys
import cv2
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
import numpy as np
import math

import tensorflow as tf
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from rl_training.msg import AngleAxis_rotation_msg
import time

import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

from std_msgs.msg import Int64
from cv_bridge import CvBridge, CvBridgeError

import abc
from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

rgb_bridge = CvBridge()
depth_bridge = CvBridge()

grab_normal_rgb_bridge = CvBridge()
grab_approach_rgb_bridge = CvBridge()
grab_open_rgb_bridge = CvBridge()

rgb_image = np.zeros((0,0,3), np.uint8)
depth_image = np.zeros((0,0,1), np.uint8)

grab_normal_rgb_image = np.zeros((0,0,3), np.uint8)
grab_approach_rgb_image = np.zeros((0,0,3), np.uint8)
grab_open_rgb_image = np.zeros((0,0,3), np.uint8)

number_of_grab_pointClouds = 0

xyz = np.array([[0,0,0]])
rgb = np.array([[0,0,0]])

def grab_pointClouds_callback(ros_point_cloud):
    global xyz, rgb
    #self.lock.acquire()
    gen = pc2.read_points(ros_point_cloud, skip_nans=True)
    int_data = list(gen)

    xyz = np.array([[0,0,0]])
    rgb = np.array([[0,0,0]])
    
    for x in int_data:
        test = x[3] 
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,test)
        i = struct.unpack('>l',s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)
        # prints r,g,b values in the 0-255 range
                    # x,y,z can be retrieved from the x[0],x[1],x[2]
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
        rgb = np.append(rgb,[[r,g,b]], axis = 0)

    # rospy.loginfo('Done grab_pointClouds_callback')

def rgb_callback(image):
    global rgb_image
    try:
        rgb_image = rgb_bridge.imgmsg_to_cv2(image, "bgr8")
        # cv2.namedWindow('rgb_image', cv2.WINDOW_NORMAL)
        # cv2.imshow('rgb_image', rgb_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)

def depth_callback(image):
    global depth_image
    try:
        depth_image = depth_bridge.imgmsg_to_cv2(image)
        # cv2.namedWindow('depth_image', cv2.WINDOW_NORMAL)
        # cv2.imshow('depth_image', depth_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)

def number_of_grab_pointClouds_callback(num):
    global number_of_grab_pointClouds
    number_of_grab_pointClouds = num
    print("number_of_grab_pointClouds: ", number_of_grab_pointClouds)

def grab_normal_rgb_callback(image):
    global grab_normal_rgb_image
    try:
        grab_normal_rgb_image = grab_normal_rgb_bridge.imgmsg_to_cv2(image, "bgr8")
        # cv2.namedWindow('grab_normal_rgb_image', cv2.WINDOW_NORMAL)
        # cv2.imshow('grab_normal_rgb_image', grab_normal_rgb_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)

def grab_approach_rgb_callback(image):
    global grab_approach_rgb_image
    try:
        grab_approach_rgb_image = grab_approach_rgb_bridge.imgmsg_to_cv2(image, "bgr8")
        # cv2.namedWindow('grab_approach_rgb_image', cv2.WINDOW_NORMAL)
        # cv2.imshow('grab_approach_rgb_image', grab_approach_rgb_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)

def grab_open_rgb_callback(image):
    global grab_open_rgb_image
    try:
        grab_open_rgb_image = grab_open_rgb_bridge.imgmsg_to_cv2(image, "bgr8")
        # cv2.namedWindow('grab_open_rgb_image', cv2.WINDOW_NORMAL)
        # cv2.imshow('grab_open_rgb_image', grab_open_rgb_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)

if __name__ == '__main__':

    rospy.init_node('Reinforcement_Learning_Trining', anonymous=True)

    # Create ROS subscriber for gripper working area pointcloud
    rospy.Subscriber("/Grab_PointClouds", PointCloud2, grab_pointClouds_callback, buff_size=52428800)

    # Create ROS subscriber for mapping rgb image from Azure input pointcloud
    rospy.Subscriber("/projected_image/rgb", Image, rgb_callback)

    # Create ROS subscriber for mapping rgb image from Azure input pointcloud
    rospy.Subscriber("/projected_image/depth", Image, depth_callback)
    
    # Create ROS subscriber for number of pointcloud in gripper working area (the reward of reinforcement learning agent...?)
    rospy.Subscriber("/Number_of_Grab_PointClouds", Int64, number_of_grab_pointClouds_callback)

    # Create ROS subscriber for mapping rgb image from gripper axis of normal vector (the state of reinforcement learning agent)
    rospy.Subscriber("/projected_image/grab_normal_rgb", Image, grab_normal_rgb_callback)

    # Create ROS subscriber for mapping rgb image from gripper axis of approach vector (the state of reinforcement learning agent)
    rospy.Subscriber("/projected_image/grab_approach_rgb", Image, grab_approach_rgb_callback)

    # Create ROS subscriber for mapping rgb image from gripper axis of open vector (the state of reinforcement learning agent)
    rospy.Subscriber("/projected_image/grab_open_rgb", Image, grab_open_rgb_callback)

    # Create ROS publisher for rotate gripper axis of normal, approach and open vector (the actions of reinforcement learning agent)
    pub_AngleAxisRotation = rospy.Publisher('/grasp_training/AngleAxis_rotation', AngleAxis_rotation_msg, queue_size=10)

    rotation = AngleAxis_rotation_msg()
    rotation_angle = math.pi/2
    interval = 100

    while not rospy.is_shutdown():
        for i in range(interval):
            rotation.x = 0
            rotation.y = 0
            rotation.z = 1*rotation_angle/interval*i
            pub_AngleAxisRotation.publish(rotation)
            time.sleep(0.1)
