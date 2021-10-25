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
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network

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

rgb_image = np.zeros((0,0,3), np.uint8)
depth_image = np.zeros((0,0,1), np.uint8)

number_of_grab_pointClouds = 0.0

xyz = np.array([[0,0,0]])
rgb = np.array([[0,0,0]])

def rotate_grasp():
    rotation = AngleAxis_rotation_msg()
    rotation_angle = math.pi/2
    interval = 50

    for i in range(interval):
        rotation.x = 0
        rotation.y = 0
        rotation.z = 1*rotation_angle/interval*i
        pub_AngleAxisRotation.publish(rotation)
        time.sleep(0.1)

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


class GraspEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=14, name="action")

        self._observation_spec = {  "grab_normal" : array_spec.BoundedArraySpec((480, 640, 3), dtype = np.float32, minimum=0, maximum=255),
                                    "grab_approach" : array_spec.BoundedArraySpec((480, 640, 3), dtype = np.float32, minimum=0, maximum=255)
                                    # "grab_open" : array_spec.BoundedArraySpec((480, 640, 3), dtype = np.float32, minimum=0, maximum=255)
                                    }

        self._state = { "grab_normal" : np.zeros((480, 640, 3), np.float32),
                        "grab_approach" : np.zeros((480, 640, 3), np.float32)
                        # "grab_open" : np.zeros((480, 640, 3), np.float32)
                        }
                        
        self._episode_ended = False

        self._reward = 0 

        self.grab_normal_rgb_image = np.zeros((0,0,3), np.float32)
        self.grab_approach_rgb_image = np.zeros((0,0,3), np.float32)
        self.grab_open_rgb_image = np.zeros((0,0,3), np.float32)

        # Create ROS subscriber for number of pointcloud in gripper working area (the reward of reinforcement learning agent...?)
        rospy.Subscriber("/Number_of_Grab_PointClouds", Int64, self.number_of_grab_pointClouds_callback)

        # Create ROS subscriber for mapping rgb image from gripper axis of normal vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_normal_rgb", Image, self.grab_normal_rgb_callback)

        # Create ROS subscriber for mapping rgb image from gripper axis of approach vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_approach_rgb", Image, self.grab_approach_rgb_callback)

        # Create ROS subscriber for mapping rgb image from gripper axis of open vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_open_rgb", Image, self.grab_open_rgb_callback)

    def number_of_grab_pointClouds_callback(self, num):
        self.number_of_grab_pointClouds = num.data
        # print("number_of_grab_pointClouds: ", number_of_grab_pointClouds)

    def grab_normal_rgb_callback(self, image):
        try:
            self.grab_normal_rgb_image = grab_normal_rgb_bridge.imgmsg_to_cv2(image, "bgr8").astype(np.float32)/255
            # cv2.namedWindow('grab_normal_rgb_image', cv2.WINDOW_NORMAL)
            # cv2.imshow('grab_normal_rgb_image', self.grab_normal_rgb_image)
            # cv2.waitKey(1)
            # cv2.imwrite("/home/luca-home-ubuntu20/code/RVP_GGCNN/grab_normal.jpg", self.grab_normal_rgb_image)
            # print("self.grab_normal_rgb_image: ", self.grab_normal_rgb_image)
        except CvBridgeError as e:
            print(e)

    def grab_approach_rgb_callback(self, image):
        try:
            self.grab_approach_rgb_image = grab_approach_rgb_bridge.imgmsg_to_cv2(image, "bgr8").astype(np.float32)/255
            # cv2.namedWindow('grab_approach_rgb_image', cv2.WINDOW_NORMAL)
            # cv2.imshow('grab_approach_rgb_image', grab_approach_rgb_image)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

    def grab_open_rgb_callback(self, image):
        try:
            self.grab_open_rgb_image = grab_open_rgb_bridge.imgmsg_to_cv2(image, "bgr8").astype(np.float32)/255
            # cv2.namedWindow('grab_open_rgb_image', cv2.WINDOW_NORMAL)
            # cv2.imshow('grab_open_rgb_image', grab_open_rgb_image)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._state = { "grab_normal" : np.zeros((480, 640, 3), np.float32),
                        "grab_approach" : np.zeros((480, 640, 3), np.float32)
                        # "grab_open" : np.zeros((480, 640, 3), np.float32)
                        }
        self._reward = 0 
        self._episode_ended = False
        return ts.restart(self._state)
    
    def _update_ROS_data(self):
        self._state["grab_normal"] = self.grab_normal_rgb_image
        self._state["grab_approach"] = self.grab_approach_rgb_image
        self._state["grab_open"] = self.grab_open_rgb_image
        self._reward = self.number_of_grab_pointClouds

    def _step(self, action):

        self._update_ROS_data()
        
        if self._episode_ended:
            return self.reset()
        
        if action[0] == 0.0 and action[1] == 0.0 and action[2] == 0.0:
            # print("environment stop!")
            self._episode_ended = True
            return ts.transition(self._state, self._reward, discount=1.0)

        else:
            return ts.transition(self._state, self._reward, discount=1.0)

class ActorNetwork(network.Network):

  def __init__(self,
               observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
               name='ActorNetwork'):

    super(ActorNetwork, self).__init__(
        input_tensor_spec=observation_spec, state_spec=(), name=name)

    # For simplicity we will only support a single action float output.
    self._action_spec = action_spec
    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 4:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]
    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')

    kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')

    self._encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)

    initializer = tf.keras.initializers.RandomUniform(
        minval=-0.003, maxval=0.003)

    self._action_projection_layer = tf.keras.layers.Dense(
        flat_action_spec[0].shape.num_elements(),
        activation=tf.keras.activations.tanh,
        kernel_initializer=initializer,
        name='action')

  def call(self, observations, step_type=(), network_state=()):
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    # We use batch_squash here in case the observations have a time sequence
    # compoment.
    batch_squash = utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state)
    actions = self._action_projection_layer(state)
    actions = common_utils.scale_to_spec(actions, self._single_action_spec)
    actions = batch_squash.unflatten(actions)
    return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state



if __name__ == '__main__':

    rospy.init_node('Reinforcement_Learning_Trining', anonymous=True)

    # Create ROS subscriber for gripper working area pointcloud
    rospy.Subscriber("/Grab_PointClouds", PointCloud2, grab_pointClouds_callback, buff_size=52428800)

    # Create ROS subscriber for mapping rgb image from Azure input pointcloud
    rospy.Subscriber("/projected_image/rgb", Image, rgb_callback)

    # Create ROS subscriber for mapping rgb image from Azure input pointcloud
    rospy.Subscriber("/projected_image/depth", Image, depth_callback)

    # Create ROS publisher for rotate gripper axis of normal, approach and open vector (the actions of reinforcement learning agent)
    pub_AngleAxisRotation = rospy.Publisher('/grasp_training/AngleAxis_rotation', AngleAxis_rotation_msg, queue_size=10)

    environment = GraspEnv()

    # env_utils.validate_py_environment(environment, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(environment)

    preprocessing_layers = {
    'grab_normal': tf.keras.models.Sequential([ tf.keras.layers.Conv2D(3, 3),
                                                tf.keras.layers.Conv2D(3, 3),
                                                tf.keras.layers.Flatten()]),

    'grab_approach': tf.keras.models.Sequential([ tf.keras.layers.Conv2D(3, 3),
                                                tf.keras.layers.Conv2D(3, 3),
                                                tf.keras.layers.Flatten()])

    # 'grab_open': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 4),
    #                                     tf.keras.layers.Flatten()]),    
                                        }

    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)


    my_q_network = tf_agents.networks.q_network.QNetwork(
                    tf_env.observation_spec(), 
                    tf_env.action_spec(), 
                    preprocessing_layers=preprocessing_layers,
                    preprocessing_combiner=preprocessing_combiner, 
                    conv_layer_params=None, 
                    fc_layer_params=(75, 40),
                    dropout_layer_params=None, 
                    activation_fn=tf.keras.activations.relu,
                    kernel_initializer=None, 
                    batch_squash=True, 
                    dtype=tf.float32,
                    name='QNetwork'
                )

    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=my_q_network,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    time_step = tf_env.reset()

    while not rospy.is_shutdown():

        print("ros is not shutdown!")
        
        # rotate_grasp()