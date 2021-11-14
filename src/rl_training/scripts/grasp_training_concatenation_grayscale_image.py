#!/usr/bin/env python3
import sys
import cv2
from tensorflow.python.ops.math_ops import truediv
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
from rl_training.msg import AngleAxis_rotation_msg
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

def rotate_grasp():
    rotation = AngleAxis_rotation_msg()
    rotation_angle = math.pi/2
    interval = 50

    for i in range(interval):
        rotation.x = 0
        rotation.y = -1*rotation_angle/interval*i
        rotation.z = 0
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

    except CvBridgeError as e:
        print(e)

def depth_callback(image):
    global depth_image
    try:
        depth_image = depth_bridge.imgmsg_to_cv2(image)

    except CvBridgeError as e:
        print(e)


class GraspEnv(py_environment.PyEnvironment):

    def __init__(self, input_image_size):
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=18, name="action")

        self.input_image_size = input_image_size

        self._observation_spec = {  "depth_grab" : array_spec.BoundedArraySpec((self.input_image_size[0], self.input_image_size[1], 3), dtype = np.float32, minimum=0, maximum=255)
                                    }

        self._state = { "depth_grab" : np.zeros((self.input_image_size[0], self.input_image_size[1], 3), np.float32)
                        }
        
        self.grab_normal_depth_image = np.zeros((0,0,1), np.float32)
        self.grab_approach_depth_image = np.zeros((0,0,1), np.float32)
        self.grab_open_depth_image = np.zeros((0,0,1), np.float32)

        self.grab_depth_image = np.zeros((0,0,3), np.float32)


        self._episode_ended = False

        self._reward = 0 
        self._step_counter = 0
        self._step_lengh = 90
        self._number_of_grab_pointClouds = 0
        self._number_of_finger_grab_pointClouds = 0
        self.pointLikelihoos_left_finger = 0
        self.pointLikelihoos_right_finger = 0
        self.apporachLikelihood = 0

        self.rotate_x = 0 
        self.rotate_y = 0 
        self.rotate_z = 0 

        # Create ROS subscriber for number of pointcloud in gripper working area (the reward of reinforcement learning agent...?)
        rospy.Subscriber("/Number_of_Grab_PointClouds", Int64, self.number_of_grab_pointClouds_callback)

        # Create ROS subscriber for mapping rgb image from gripper axis of normal vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_normal_rgb", Image, self.grab_normal_rgb_callback)

        # Create ROS subscriber for mapping rgb image from gripper axis of approach vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_approach_rgb", Image, self.grab_approach_rgb_callback)

        # Create ROS subscriber for mapping rgb image from gripper axis of open vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_open_rgb", Image, self.grab_open_rgb_callback)

        # Create ROS subscriber for mapping depth image from gripper axis of normal vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_normal_depth", Image, self.grab_normal_depth_callback)

        # Create ROS subscriber for mapping depth image from gripper axis of approach vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_approach_depth", Image, self.grab_approach_depth_callback)

        # Create ROS subscriber for mapping depth image from gripper axis of open vector (the state of reinforcement learning agent)
        rospy.Subscriber("/projected_image/grab_open_depth", Image, self.grab_open_depth_callback)

        rospy.Subscriber("/Number_of_Finger_Grab_PointClouds", Int64, self.finger_point_callback)

        rospy.Subscriber("/PointLikelihood/Left_Finger", Float64, self.pointLikelihoos_left_finger_callback)

        rospy.Subscriber("/PointLikelihood/Right_Finger", Float64, self.pointLikelihoos_right_finger_callback)

        rospy.Subscriber("/ApporachLikelihood", Float64, self.apporachLikelihood_callback)
    
    def apporachLikelihood_callback(self, num):
        self.apporachLikelihood = num.data

    def pointLikelihoos_left_finger_callback(self, num):
        self.pointLikelihoos_left_finger = num.data

    def pointLikelihoos_right_finger_callback(self, num):
        self.pointLikelihoos_right_finger = num.data

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

        self._update_ROS_data()
        self._reward = 0 
        self._episode_ended = False

        initial_angle = (math.pi*90)/180

        self.rotate_x = initial_angle * (random.random() - 0.5)
        self.rotate_y = initial_angle * (random.random() - 0.5)
        self.rotate_z = initial_angle * (random.random() - 0.5)

        return ts.restart(self._state)
    
    def _rotate_grasp(self, rotate_axis):
        
        rotation = AngleAxis_rotation_msg()
        rotation.x = self.rotate_x
        rotation.y = self.rotate_y
        rotation.z = self.rotate_z

        # 1 degree
        rotation_angle_l = math.pi/180

        rotation_angle_m = (math.pi*5)/180

        rotation_angle_b = (math.pi*10)/180


        if rotate_axis not in ["x_l", "-x_l", "y_l", "-y_l", "z_l", "-z_l", 
                               "x_m", "-x_m", "y_m", "-y_m", "z_m", "-z_m", 
                               "x_b", "-x_b", "y_b", "-y_b", "z_b", "-z_b", 
                               "stop"]:

            print("rotate_axis must in [x, y, z ,-x, -y, -z, stop]!!")
        
        else:
            if rotate_axis == "x_l":

                self.rotate_x = self.rotate_x + rotation_angle_l
                
                rotation.x = self.rotate_x

            elif rotate_axis == "-x_l":

                self.rotate_x = self.rotate_x - rotation_angle_l

                rotation.x =  self.rotate_x

            elif rotate_axis == "y_l":

                self.rotate_y = self.rotate_y + rotation_angle_l

                rotation.y = self.rotate_y 

            elif rotate_axis == "-y_l":

                self.rotate_y = self.rotate_y - rotation_angle_l

                rotation.y =  self.rotate_y 

            elif rotate_axis == "z_l":

                self.rotate_z = self.rotate_z + rotation_angle_l

                rotation.z = self.rotate_z

            elif rotate_axis == "-z_l":

                self.rotate_z = self.rotate_z - rotation_angle_l

                rotation.z = self.rotate_z

            elif rotate_axis == "x_m":

                self.rotate_x = self.rotate_x + rotation_angle_m
                
                rotation.x = self.rotate_x

            elif rotate_axis == "-x_m":

                self.rotate_x = self.rotate_x - rotation_angle_m

                rotation.x = self.rotate_x

            elif rotate_axis == "y_m":

                self.rotate_y = self.rotate_y + rotation_angle_m

                rotation.y = self.rotate_y 

            elif rotate_axis == "-y_m":

                self.rotate_y = self.rotate_y - rotation_angle_m

                rotation.y = self.rotate_y 

            elif rotate_axis == "z_m":

                self.rotate_z = self.rotate_z + rotation_angle_m

                rotation.z = self.rotate_z

            elif rotate_axis == "-z_m":

                self.rotate_z = self.rotate_z - rotation_angle_m

                rotation.z = self.rotate_z

            elif rotate_axis == "x_b":

                self.rotate_x = self.rotate_x + rotation_angle_b
                
                rotation.x = self.rotate_x

            elif rotate_axis == "-x_b":

                self.rotate_x = self.rotate_x - rotation_angle_b

                rotation.x = self.rotate_x

            elif rotate_axis == "y_b":

                self.rotate_y = self.rotate_y + rotation_angle_b

                rotation.y = self.rotate_y 

            elif rotate_axis == "-y_b":

                self.rotate_y = self.rotate_y - rotation_angle_b

                rotation.y = self.rotate_y 

            elif rotate_axis == "z_b":

                self.rotate_z = self.rotate_z + rotation_angle_b

                rotation.z = self.rotate_z

            elif rotate_axis == "-z_b":

                self.rotate_z = self.rotate_z - rotation_angle_b

                rotation.z = self.rotate_z
            
            elif rotate_axis == "stop":
                pass

            else:
                print("something wrong..")

            pub_AngleAxisRotation.publish(rotation)



    def _update_ROS_data(self):

        self._state["depth_grab"] = np.concatenate((self.grab_normal_depth_image, self.grab_approach_depth_image, self.grab_open_depth_image), axis=-1)
        # print("self._state[depth_grab].shape", self._state["depth_grab"].shape)

    def _update_reward(self):
        self._reward = 10*(self.pointLikelihoos_right_finger + self.pointLikelihoos_left_finger) + 20*(self.apporachLikelihood) #- self._step_counter

    def _step(self, action):
        time.sleep(0.012)
        self._update_ROS_data()
        self._update_reward()

        if self._episode_ended:
            return self.reset()
        
        if action == 0:

            self._rotate_grasp("x_l")

        elif action ==1:

            self._rotate_grasp("-x_l")

        elif action ==2:

            self._rotate_grasp("y_l")

        elif action ==3:

            self._rotate_grasp("-y_l")

        elif action ==4:

            self._rotate_grasp("z_l")

        elif action ==5:

            self._rotate_grasp("-z_l")
        
        elif action ==6:

            self._rotate_grasp("x_m")

        elif action ==7:

            self._rotate_grasp("-x_m")

        elif action ==8:

            self._rotate_grasp("y_m")

        elif action ==9:

            self._rotate_grasp("-y_m")

        elif action ==10:

            self._rotate_grasp("z_m")

        elif action ==11:

            self._rotate_grasp("-z_m")

        elif action ==12:

            self._rotate_grasp("x_b")

        elif action ==13:

            self._rotate_grasp("-x_b")

        elif action ==14:

            self._rotate_grasp("y_b")

        elif action ==15:

            self._rotate_grasp("-y_b")

        elif action ==16:

            self._rotate_grasp("z_b")

        elif action ==17:

            self._rotate_grasp("-z_b")
        
        elif action ==18:

            self._rotate_grasp("stop")
            self._episode_ended = True
            self._step_counter = 0
            print("stop!")

        else:

            self._episode_ended = True
        
        self._step_counter = self._step_counter +1
        
        if self._number_of_finger_grab_pointClouds > 0:
            self._episode_ended = True
            self._step_counter = 0
            return ts.termination(self._state, -1000)

        if (abs(self.rotate_z) > (math.pi*45)/180) or (abs(self.rotate_x) > (math.pi*45)/180) or (abs(self.rotate_y) > (math.pi*45)/180):
            self._episode_ended = True
            self._step_counter = 0
            # print("self.rotate_x ", self.rotate_x, ", self.rotate_y ", self.rotate_y, ", self.rotate_z ", self.rotate_z)
            return ts.termination(self._state, -1000)

        if self._step_counter > self._step_lengh:
            self._episode_ended = True
            self._step_counter = 0
            return ts.termination(self._state, self._reward)

        else:
            return ts.transition(self._state, self._reward, discount=1.0)


if __name__ == '__main__':

    rospy.init_node('Reinforcement_Learning_Trining', anonymous=True)

    # # Create ROS subscriber for gripper working area pointcloud
    # rospy.Subscriber("/Grab_PointClouds", PointCloud2, grab_pointClouds_callback, buff_size=52428800)

    # # Create ROS subscriber for mapping rgb image from Azure input pointcloud
    # rospy.Subscriber("/projected_image/rgb", Image, rgb_callback)

    # # Create ROS subscriber for mapping rgb image from Azure input pointcloud
    # rospy.Subscriber("/projected_image/depth", Image, depth_callback)

    # Create ROS publisher for rotate gripper axis of normal, approach and open vector (the actions of reinforcement learning agent)
    pub_AngleAxisRotation = rospy.Publisher('/grasp_training/AngleAxis_rotation', AngleAxis_rotation_msg, queue_size=10)

    environment = GraspEnv([120, 160])

    time.sleep(1)

    env_utils.validate_py_environment(environment, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(environment)

    
    preprocessing_layers = {
    'depth_grab': tf.keras.models.Sequential([ 
        tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten()])
                                        }

    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)


    my_q_network = tf_agents.networks.q_network.QNetwork(
                    tf_env.observation_spec(), 
                    tf_env.action_spec(), 
                    preprocessing_layers=preprocessing_layers,
                    conv_layer_params=None, 
                    fc_layer_params=(100, 100),
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

    print("my_q_network.summary(): ", my_q_network.summary())

    replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                                                            batch_size=tf_env.batch_size,
                                                                                            max_length=64*40)

    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        time_start = time.time()

        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        print("avg execute time ", (time.time()-time_start)/num_episodes)

        avg_return = total_return / num_episodes

        return avg_return.numpy()[0]

    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        # print("next_time_step.reward ", next_time_step.reward)
        traj = tf_agents.trajectories.trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
    
    avg_return = compute_avg_return(tf_env, agent.policy, 5)
    returns = [avg_return]

    collect_steps_per_iteration = 1
    batch_size = 64
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, 
                                        sample_batch_size=batch_size, 
                                        num_steps=2).prefetch(3)
    iterator = iter(dataset)
    num_iterations = 10000

    time_step = tf_env.reset()

    TRAIN_LOSS = []
    AVG_RETURN = []
    STEP = []

    train_loss_file = "/home/luca-home-ubuntu20/code/RVP_GGCNN/src/rl_training/scripts/training_result/TRAIN_LOSS.pkl"
    avf_return_file = "/home/luca-home-ubuntu20/code/RVP_GGCNN/src/rl_training/scripts/training_result/AVG_RETURN.pkl"
    step_file = "/home/luca-home-ubuntu20/code/RVP_GGCNN/src/rl_training/scripts/training_result/STEP.pkl"


    while not rospy.is_shutdown():

        for _ in range(batch_size):
            collect_step(tf_env, agent.policy, replay_buffer)

        for _ in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                collect_step(tf_env, agent.collect_policy, replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            # Print loss every 200 steps.
            if step % 200 == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))
                STEP.append(step)
                TRAIN_LOSS.append(train_loss.numpy())

                open_file = open(train_loss_file, "wb")
                pickle.dump(TRAIN_LOSS, open_file)
                open_file.close()

                open_file = open(step_file, "wb")
                pickle.dump(STEP, open_file)
                open_file.close()

            # Evaluate agent's performance every 1000 steps.
            if step % 1000 == 0:
                avg_return = compute_avg_return(tf_env, agent.policy, 5)

                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
                AVG_RETURN.append(avg_return)

                open_file = open(avf_return_file, "wb")
                pickle.dump(AVG_RETURN, open_file)
                open_file.close()