# OD-GraspNet-RL
### We present a 6-DoF robotic grasping method based on OD-GraspNet and Reinforcement Learning method (C51).
# Usage

### This method using **UR5 robot** with **Robotiq 2f-85 gripper** and **Microsoft Azure Kinect DK**

## Hardware 

### ur5
```
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.12

roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch
```
### azure kinect dk
```
roslaunch azure_kinect_ros_driver multi_device_driver_top.launch
```
### 2f-85 gripper
```
sudo chmod 777 /dev/ttyUSB0

roslaunch robotiq_2f_gripper_control robotiq_2f_gripper_RtuNode.launch comport:=/dev/ttyUSB0

rosrun robotiq_2f_gripper_control Robotiq2FGripperSimpleController.py
```
## ROS Command

### rviz
```
rosrun rviz rviz -d src/pcl_utils/rviz/my_rviz.rviz 
```
### pcl handle
```
rosrun pcl_utils pcl_service
```
### cloud alignment
```
rosrun pcl_utils cloud_alignment
```
### 2d grasp predict
```
rosrun dl_grasp run_realtime.py
```
### 3d grasp predict
```
rosrun rl_grasp run_agent.py 
```
### convert 2d & 3d predict to robot baselink
```
rosrun ur_move position_coverter.py
```
### ur5 moveit! 
```
rosrun ur_move ur_strategy.py 
```
# Hand eye calibration

### If need, use this command to do hand eye calibration
```
roslaunch charuco_detector ur5_eye_to_hand.launch 
```
# Training RL Grasp

### roscore
```
roscore
```
### rviz
```
rosrun rviz rviz -d src/pcl_utils/rviz/my_rviz.rviz 
```
### sample q image
```
rosrun dl_grasp sample_q_image.py
```
### pcl handle
```
rosrun pcl_utils pcl_service
```
### cloud alignment
```
rosrun pcl_utils cloud_alignment
```
### rl training
```
rosrun rl_training grasp_training_dqn.py
```

### if need, get rl training data
```
rosrun pcl_utils get_cornell_data 99
```
