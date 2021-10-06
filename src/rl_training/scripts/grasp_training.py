#!/usr/bin/env python3
import sys
import cv2
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
import numpy as np

import tensorflow as tf
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from rl_training.msg import AngleAxis_rotation_msg

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

rgb_image = np.zeros((0,0,3), np.uint8)
depth_image = np.zeros((0,0,1), np.uint8)

number_of_grab_pointClouds = 0

xyz = np.array([[0,0,0]])
rgb = np.array([[0,0,0]])

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
    if len(flat_action_spec) > 1:
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

    pub_AngleAxisRotation = rospy.Publisher('/grasp_training/AngleAxis_rotation', AngleAxis_rotation_msg, queue_size=10)

    rotation = AngleAxis_rotation_msg()
    rotation.rotation_open = 0
    rotation.rotation_approach = 0
    rotation.rotation_normal = 0

    while not rospy.is_shutdown():
        pub_AngleAxisRotation.publish(rotation)