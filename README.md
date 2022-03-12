#ur5
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.12

roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch

#azure
roslaunch azure_kinect_ros_driver multi_device_driver_top.launch

#rviz
rosrun rviz rviz -d src/pcl_utils/rviz/my_rviz.rviz 

#cloud alignment
rosrun pcl_utils cloud_alignment

#2f-85
sudo chmod 777 /dev/ttyUSB0

roslaunch robotiq_2f_gripper_control robotiq_2f_gripper_RtuNode.launch comport:=/dev/ttyUSB0

rosrun robotiq_2f_gripper_control Robotiq2FGripperSimpleController.py

#pcl handle
rosrun pcl_utils pcl_service

#2d grasp predict
rosrun dl_grasp run_realtime.py

#3d grasp predict
rosrun rl_grasp run_agent.py 

#ur5 moveit! 
rosrun ur_move ur_strategy.py 

# hand eye calibration
roslaunch charuco_detector ur5_eye_to_hand.launch 

