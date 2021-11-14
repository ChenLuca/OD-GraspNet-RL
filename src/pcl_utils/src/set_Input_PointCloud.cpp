#include <ros/ros.h>
#include "pcl_utils/setPointCloud.h"

using namespace std;

int main (int argc, char** argv)
{
  ros::init(argc, argv, "set_Input_PointCloud_Clien");
  ros::NodeHandle nh;
  ros::ServiceClient set_Input_PointCloud_clien = nh.serviceClient<pcl_utils::setPointCloud>("set_pointcloud");
  pcl_utils::setPointCloud setPointCloud;

  setPointCloud.request.call = 1;
  set_Input_PointCloud_clien.call(setPointCloud);

  return 0;
}