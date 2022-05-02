#!/usr/bin/env python3
import sys
import cv2
import os

from tensorflow.python.ops.math_ops import truediv
from yaml.loader import Loader
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
from skimage.filters import gaussian
import rospy
import numpy as np
import math
import pickle
import random 
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
from rl_training.srv import loadPointCloud
from rl_training.srv import get_RL_Env
import time

import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

from std_msgs.msg import Int64, Float64
from cv_bridge import CvBridge, CvBridgeError
from pcl_utils.msg import RL_Env_msg

import abc
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils as env_utils
from tf_agents.environments import wrappers
from tf_agents.environments import random_py_environment

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

from tf_agents.networks import network
from tf_agents.networks import encoding_network
from tf_agents.networks import utils

from tf_agents.utils import common
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

rgb_bridge = CvBridge()
depth_bridge = CvBridge()

grab_normal_rgb_bridge = CvBridge()
grab_approach_rgb_bridge = CvBridge()
grab_open_rgb_bridge = CvBridge()

grab_normal_depth_bridge = CvBridge()
grab_approach_depth_bridge = CvBridge()
grab_open_depth_bridge = CvBridge()

rgb_image = np.zeros((0,0,3), np.uint8)
depth_image = np.zeros((0,0,1), np.uint8)

xyz = np.array([[0,0,0]])
rgb = np.array([[0,0,0]])


grab_normal_depth_bridge = CvBridge()
grab_approach_depth_bridge = CvBridge()
grab_open_depth_bridge = CvBridge()
rgb_bridge = CvBridge()
depth_bridge = CvBridge()

grab_normal_rgb_bridge = CvBridge()
grab_approach_rgb_bridge = CvBridge()
grab_open_rgb_bridge = CvBridge()

class GraspEnv(py_environment.PyEnvironment):

    def __init__(self, input_image_size, phase):
        
        # must be odd number
        self.num_actions = 49

        self.input_image_size = input_image_size

        self.input_channel = 2

        self._step_lengh = 1

        print("!!!!!!!!!!!!!!!!!!!!self._step_lengh: ", self._step_lengh)

        self.phase = phase

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=(self.num_actions - 1), name="action")

        self._observation_spec = {"depth_grab" : array_spec.BoundedArraySpec((self.input_image_size[0], self.input_image_size[1], self.input_channel), dtype = np.float32, minimum=0, maximum=255)}

        self._state = {"depth_grab" : np.zeros((self.input_image_size[0], self.input_image_size[1], self.input_channel), np.float32)}
        
        self.grab_normal_depth_image = np.zeros((0,0,1), np.float32)
        self.grab_approach_depth_image = np.zeros((0,0,1), np.float32)
        self.grab_open_depth_image = np.zeros((0,0,1), np.float32)

        self.grab_depth_image = np.zeros((0,0,self.input_channel), np.float32)

        self._episode_ended = False

        self._reward = 0 
        self._step_counter = 0

        self._number_of_grab_pointClouds = 0
        self._number_of_finger_grab_pointClouds = 0
        self.pointLikelihood_left_finger = 0
        self.pointLikelihood_right_finger = 0
        self.pointLikelihood_grab_cloud = 0
        self.apporachLikelihood = 0
        self.NormalDepthNonZero =0
        self.OpenDepthNonZero =0
        self.principal_curvatures_gaussian = 0
        self.approach_mean = 0
        self.approach_stddev = 0
        self.normal_mean = 0
        self.normal_stddev = 0

        self.Maxprincipal_curvatures_gaussian = 0.0001
        self.MaxNormalDepthNonZero = 3000
        self.MaxOpenDepthNonZero = 1
        self.Max_number_of_grab_pointClouds = 1
        self.Maxapproach_mean = 20
        self.Maxapproach_stddev = 40

        self.action_stop = False
        self.rotate_x = 0 
        self.rotate_y = 0 
        self.rotate_z = 0 
        
        self.handle_get_RL_Env = rospy.ServiceProxy('/get_RL_Env', get_RL_Env)

        # Create ROS publisher for rotate gripper axis of normal, approach and open vector (the actions of reinforcement learning agent)
        self.pub_AngleAxisRotation = rospy.Publisher('/grasp_training/AngleAxis_rotation', AngleAxis_rotation_msg, queue_size=10)

    def get_RL_Env_data(self, req):
        # print("Request RL Env !")
        rospy.wait_for_service('/get_RL_Env')
        try:
            res = self.handle_get_RL_Env(req)

            if (res.state.grab_normal_depth_msg.height == 0 or res.state.grab_normal_depth_msg.width ==0):
                self.grab_normal_depth_image = np.zeros((120,160,1), np.float32)
            else:
                self.grab_normal_depth_image = gaussian(np.expand_dims(grab_normal_depth_bridge.imgmsg_to_cv2(res.state.grab_normal_depth_msg, "mono8").astype(np.float32)/255, axis =-1), 2.0, preserve_range=True)

            if (res.state.grab_open_depth_msg.height == 0 or res.state.grab_open_depth_msg.width == 0):
                self.grab_open_depth_image =np.zeros((120,160,1), np.float32)
            else:
                self.grab_open_depth_image = gaussian(np.expand_dims(grab_normal_depth_bridge.imgmsg_to_cv2(res.state.grab_open_depth_msg, "mono8").astype(np.float32)/255, axis =-1), 2.0, preserve_range=True)

            if (res.state.grab_approach_depth_msg.height == 0 or res.state.grab_approach_depth_msg.width == 0):
                self.grab_approach_depth_image = np.zeros((120,160,1), np.float32)
            else:
                self.grab_approach_depth_image = gaussian(np.expand_dims(grab_normal_depth_bridge.imgmsg_to_cv2(res.state.grab_approach_depth_msg, "mono8").astype(np.float32)/255, axis =-1), 2.0, preserve_range=True)

            self.apporachLikelihood = res.state.approach_likelihood_msg
            self.pointLikelihood_right_finger = res.state.right_likelihood_msg
            self.pointLikelihood_left_finger = res.state.left_likelihood_msg
            self.NormalDepthNonZero = res.state.NormaldepthNonZeroValue_msg
            self.pointLikelihood_grab_cloud = res.state.normal_likelihood_msg
            self._number_of_grab_pointClouds = res.state.grab_point_num
            self._number_of_finger_grab_pointClouds = res.state.finger_grab_point_num
            self.approach_mean = res.state.approach_mean
            self.approach_stddev = res.state.approach_stddev
            self.normal_mean = res.state.normal_mean
            self.normal_stddev = res.state.normal_stddev

            if math.isnan(res.state.principal_curvatures_gaussian_msg):
                self.principal_curvatures_gaussian = 0
            else:
                self.principal_curvatures_gaussian = res.state.principal_curvatures_gaussian_msg

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def pointLikelihoos_grab_cloud_callback(self, num):
        self.pointLikelihood_grab_cloud = num.data

    def OpenDepthNonZero_callback(self, num):
        self.OpenDepthNonZero = num.data

    def NormalDepthNonZero_callback(self, num):
        self.NormalDepthNonZero = num.data
    
    def apporachLikelihood_callback(self, num):
        self.apporachLikelihood = num.data

    def pointLikelihood_left_finger_callback(self, num):
        self.pointLikelihood_left_finger = num.data

    def pointLikelihood_right_finger_callback(self, num):
        self.pointLikelihood_right_finger = num.data

    def finger_point_callback(self, num):
        self._number_of_finger_grab_pointClouds = num.data

    def number_of_grab_pointClouds_callback(self, num):
        self._number_of_grab_pointClouds = num.data

    def grab_normal_rgb_callback(self, image):
        try:
            self.grab_normal_rgb_image = grab_normal_rgb_bridge.imgmsg_to_cv2(image, "bgr8").astype(np.float32)/255

        except CvBridgeError as e:
            print(e)

    def grab_approach_rgb_callback(self, image):
        try:
            self.grab_approach_rgb_image = grab_approach_rgb_bridge.imgmsg_to_cv2(image, "bgr8").astype(np.float32)/255

        except CvBridgeError as e:
            print(e)

    def grab_open_rgb_callback(self, image):
        try:
            self.grab_open_rgb_image = grab_open_rgb_bridge.imgmsg_to_cv2(image, "bgr8").astype(np.float32)/255

        except CvBridgeError as e:
            print(e)

    def grab_normal_depth_callback(self, image):
        try:
            self.grab_normal_depth_image = np.expand_dims(grab_normal_depth_bridge.imgmsg_to_cv2(image, "mono8").astype(np.float32)/255, axis =-1)
            grab_normal_depth_image_gaussian = gaussian(self.grab_normal_depth_image, 2.0, preserve_range=True)
        except CvBridgeError as e:
            print(e)

    def grab_approach_depth_callback(self, image):
        try:
            self.grab_approach_depth_image = np.expand_dims(grab_approach_depth_bridge.imgmsg_to_cv2(image, "mono8").astype(np.float32)/255, axis=-1)

        except CvBridgeError as e:
            print(e)

    def grab_open_depth_callback(self, image):
        try:
            self.grab_open_depth_image = np.expand_dims(grab_open_depth_bridge.imgmsg_to_cv2(image, "mono8").astype(np.float32)/255, axis=-1)

        except CvBridgeError as e:
            print(e)

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):

        self._reward = 0 
        self._episode_ended = False

        rotation = AngleAxis_rotation_msg()

        self.rotate_x = 0
        self.rotate_y = 0
        self.rotate_z = 0

        rotation.x = self.rotate_x
        rotation.y = self.rotate_y
        rotation.z = self.rotate_z

        self.pub_AngleAxisRotation.publish(rotation)

        # time.sleep(0.025)
        self._update_ROS_data()
        # print("reset!")
        return ts.restart(self._state)

    def _set_action(self, num_actions, inpt_action):
        inpt_action = inpt_action
        
        num_actions_sqrt = int(math.sqrt(num_actions))

        x = int(inpt_action/num_actions_sqrt) - math.floor(num_actions_sqrt/2)
        y = int(inpt_action%num_actions_sqrt) - math.floor(num_actions_sqrt/2)

        return x , y

    def _rotate_grasp(self, action_value):
        
        rotation = AngleAxis_rotation_msg()
        rotation.x = 0
        rotation.y = 0
        rotation.z = 0

        # 1 degree
        rotation_angle_s = math.pi/180

        # 10 degree
        rotation_angle_m = (math.pi*5)/180

        # 20  degree
        rotation_angle_l = (math.pi*20)/180 

        # 10 degree
        rotation_angle_10 = (math.pi*10)/180 

        # 15 degree
        rotation_angle_15 = (math.pi*10)/180 

        rotation_angle_x, rotation_angle_y = self._set_action(self.num_actions, action_value)

        self.rotate_x = self.rotate_x + (rotation_angle_x * rotation_angle_15)
        self.rotate_y = self.rotate_y + (rotation_angle_y * rotation_angle_15)

        rotation.x = self.rotate_x
        rotation.y = self.rotate_y

        self.pub_AngleAxisRotation.publish(rotation)

    def _update_ROS_data(self):

        # start_time = time.time()
        self.get_RL_Env_data(1)
        # print("--- %s seconds ---" % (time.time() - start_time))

        # self._state["depth_grab"] = np.concatenate((self.grab_normal_depth_image, self.grab_approach_depth_image, self.grab_open_depth_image), axis=-1)
        self._state["depth_grab"] = np.concatenate((self.grab_normal_depth_image, self.grab_approach_depth_image), axis=-1)

        # self._state["depth_grab"] = self.grab_normal_depth_image

        self._update_reward()

    def _update_reward(self):
        # if self.principal_curvatures_gaussian > self.Maxprincipal_curvatures_gaussian:
        #     self.Maxprincipal_curvatures_gaussian = self.principal_curvatures_gaussian

        # if self.OpenDepthNonZero >  self.MaxOpenDepthNonZero:
        #     self.MaxOpenDepthNonZero = self.OpenDepthNonZero

        # if self._number_of_grab_pointClouds > self.Max_number_of_grab_pointClouds:
        #     self.Max_number_of_grab_pointClouds = self._number_of_grab_pointClouds

        if self.NormalDepthNonZero >  self.MaxNormalDepthNonZero:
            self.MaxNormalDepthNonZero = self.NormalDepthNonZero

        if self.approach_mean > self.Maxapproach_mean:
            self.Maxapproach_mean = self.approach_mean

        if self.approach_stddev > self.Maxapproach_stddev:
            self.Maxapproach_stddev = self.approach_stddev

        self._reward =  - 1.0*(self.NormalDepthNonZero/self.MaxNormalDepthNonZero) \
                        + (self.pointLikelihood_right_finger) \
                        + (self.approach_mean/self.Maxapproach_mean) \
                        + (self.approach_stddev/self.Maxapproach_stddev) \
                        - (self._step_counter)*0.1 \
                        + 1.0*(self.apporachLikelihood) \
                        + self.pointLikelihood_grab_cloud

                        # + self.pointLikelihood_left_finger)
                        # + (self.principal_curvatures_gaussian) 
                        # + 1.0*(self._number_of_grab_pointClouds/self.Max_number_of_grab_pointClouds) 
                        # + 1.0*(self.OpenDepthNonZero/self.MaxOpenDepthNonZero) 

    def _step(self, action):

        if self._episode_ended:
            return self.reset()
        
        # print("action: ", action)
        
        #action!
        self._rotate_grasp(action)

        # time.sleep(0.025)

        self._update_ROS_data()
        # self._update_reward()
        self._step_counter = self._step_counter +1

        if self._number_of_finger_grab_pointClouds > 0:
            self._episode_ended = True
            self._step_counter = 0
            # print("finger crash!")
            return ts.termination(self._state, -30)

        if (abs(self.rotate_x) > (math.pi*60)/180) or (abs(self.rotate_y) > (math.pi*60)/180):
            self._episode_ended = True
            self._step_counter = 0
            print("out of angle!")
            return ts.termination(self._state, -30)

        if self.action_stop:
            self.action_stop = False
            self._episode_ended = True
            self._step_counter = 0
            print("action stop!")
            return ts.termination(self._state, 0.0)

        if self.phase == "training":
            if self._step_counter > self._step_lengh:
                self._episode_ended = True
                self._step_counter = 0
                # print("out of step!")
                return ts.termination(self._state, self._reward)

            else:
                return ts.transition(self._state, self._reward, discount=1.0)
        else:
            return ts.transition(self._state, self._reward, discount=1.0)

