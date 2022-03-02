# run .rviz
```sh
rosrun rviz rviz -d src/pcl_utils/rviz/my_rviz.rviz 
```

#ur5
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.12

roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch

# hand eye calibration
roslaunch charuco_detector ur5_eye_to_hand.launch 

