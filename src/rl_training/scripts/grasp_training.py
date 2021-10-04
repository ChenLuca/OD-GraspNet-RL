#!/usr/bin/env python3
import sys
import cv2
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
import numpy as np

import tensorflow as tf
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

from std_msgs.msg import Int64
from cv_bridge import CvBridge, CvBridgeError

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

rgb_bridge = CvBridge()
depth_bridge = CvBridge()

rgb_image = np.zeros((0,0,3), np.uint8)
depth_image = np.zeros((0,0,1), np.uint8)

number_of_grab_pointClouds = 0

xyz = np.array([[0,0,0]])
rgb = np.array([[0,0,0]])

class GraspEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))
    
    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

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

    rospy.loginfo('Done grab_pointClouds_callback')

def rgb_callback(image):
    global rgb_image
    try:
        rgb_image = rgb_bridge.imgmsg_to_cv2(image, "rgb8")
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
    # print("number_of_grab_pointClouds: ", number_of_grab_pointClouds)
    
if __name__ == '__main__':

    rospy.init_node('Reinforcement_Learning_Trining', anonymous=True)

    rospy.Subscriber("/Grab_PointClouds", PointCloud2, grab_pointClouds_callback, buff_size=52428800)
    rospy.Subscriber("/projected_image/rgb", Image, rgb_callback)
    rospy.Subscriber("/projected_image/depth", Image, depth_callback)
    rospy.Subscriber("/Number_of_Grab_PointClouds", Int64, number_of_grab_pointClouds_callback)

    environment = GraspEnv()
    utils.validate_py_environment(environment, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(environment)

    while not rospy.is_shutdown():
        pass