#!/usr/bin/env python3
import sys
import cv2
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
import numpy as np
import math

import tensorflow as tf

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

solve_cudnn_error()

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
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

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
    # print("number_of_grab_pointClouds: ", number_of_grab_pointClouds)

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

class GraspEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name="action")

        self._observation_spec = {"grab_normal":array_spec.BoundedArraySpec((640, 480, 3), dtype = np.float32, minimum=0, maximum=255),
                                    "grab_approach":array_spec.BoundedArraySpec((640, 480, 3), dtype = np.float32, minimum=0, maximum=255),
                                    "grab_open":array_spec.BoundedArraySpec((640, 480, 3), dtype = np.float32, minimum=0, maximum=255)}

        # self._observation_spec = array_spec.BoundedArraySpec(shape=(640, 480, 3), dtype=np.float32, minimum=0, maximum=255, name='observation')

        # self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, name='observation')
        # self._state = np.zeros((640,480,3), np.float32)

        self._state = {"grab_normal":np.zeros((640, 480, 3), np.float32),
                        "grab_approach":np.zeros((640, 480, 3), np.float32),
                        "grab_open":np.zeros((640, 480, 3), np.float32)}
                        
        self._episode_ended = False
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._state = {"grab_normal":np.zeros((640, 480, 3), np.float32),
                        "grab_approach":np.zeros((640, 480, 3), np.float32),
                        "grab_open":np.zeros((640, 480, 3), np.float32)}
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        
        if self._episode_ended:
            return self.reset()
        
        if action == 0:
            reward = 1.0
            self._episode_ended = True
            return ts.termination(self._state, reward)

        elif action == 1:
            reward = 2.0
            return ts.transition(self._state, reward, discount=1.0)
        else:
            raise ValueError("action should be 0 or 1!")

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

    environment = GraspEnv()
    utils.validate_py_environment(environment, episodes=5)


    while not rospy.is_shutdown():
        # for i in range(interval):
        #     rotation.x = 0
        #     rotation.y = 0
        #     rotation.z = 1*rotation_angle/interval*i
        #     pub_AngleAxisRotation.publish(rotation)
        #     time.sleep(0.1)
        time_step = environment.reset()
        print(time_step.observation)

