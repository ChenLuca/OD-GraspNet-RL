#!/usr/bin/env python3
import sys
import cv2
import os

from tensorflow.python.ops.math_ops import truediv
from yaml.loader import Loader
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
import numpy as np
import math
import pickle
import random 
import tensorflow as tf
import time

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

from rl_grasp.srv import loadPointCloud
import time

import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

from std_msgs.msg import Int64, Float64
from cv_bridge import CvBridge, CvBridgeError

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

from grasp_env_absAction import GraspEnv

if __name__ == '__main__':

    rospy.init_node('Reload_Reinforcement_Learning_Agent', anonymous=True)

    handle_loadPointCloud = rospy.ServiceProxy('/load_pointcloud', loadPointCloud)
    
    environment = GraspEnv([120, 160], "inference")

    time.sleep(1)

    # env_utils.validate_py_environment(environment, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(environment)

    # baseline DQN agent
    # policy_dir = os.path.join("./src/rl_grasp/scripts/trained-model/", 'DQN_baseline/20220306_DQN_policy_83.0_1.1929423')

    # # principal_curvatures 
    # # not stable
    # policy_dir = os.path.join("./src/rl_grasp/scripts/trained-model/", 'DQN_with_principal_curvatures_20220410/DQN_policy_42.0_76.079285')

    # # principal_curvatures
    # policy_dir = os.path.join("./src/rl_grasp/scripts/trained-model/", 'DQN_with_principal_curvatures_20220410/DQN_policy_301.0_59.550373')


    # # principal_curvatures gaussian input
    # policy_dir = os.path.join("./src/rl_grasp/scripts/trained-model/", 'DQN_principal_curvatures_gaussian_input_image_20220414/DQN_policy_180.0_71.76671')

    # # principal_curvatures gaussian input
    # policy_dir = os.path.join("./src/rl_grasp/scripts/trained-model/", 'DQN_principal_curvatures_gaussian_input_image_20220414/DQN_policy_200.0_56.337597')


    # q-sample
    policy_dir = os.path.join("./src/rl_grasp/scripts/trained-model/", 'DQN_q_sample_20220419/DQN_policy_1010.0_14.272098')

    saved_policy = tf.saved_model.load(policy_dir)

    while 1:
      time_step = tf_env.reset()
      total_reward = 0
      while not time_step.is_last():
        action_step = saved_policy.action(time_step)
        time_step = tf_env.step(action_step.action)
        total_reward += time_step.reward.numpy()
        print("time_step reward: ", time_step.reward.numpy())
      print("total_reward: ", total_reward)