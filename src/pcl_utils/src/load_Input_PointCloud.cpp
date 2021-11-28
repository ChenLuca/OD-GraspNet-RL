#include <ros/ros.h>
#include "pcl_utils/loadPointCloud.h"

using namespace std;

int main (int argc, char** argv)
{
  ros::init(argc, argv, "load_Input_PointCloud_Clien");
  ros::NodeHandle nh;
  ros::ServiceClient load_Input_PointCloud_clien = nh.serviceClient<pcl_utils::loadPointCloud>("load_pointcloud");
  pcl_utils::loadPointCloud loadPointCloud;

  loadPointCloud.request.call = 1;
  load_Input_PointCloud_clien.call(loadPointCloud);

  return 0;
}