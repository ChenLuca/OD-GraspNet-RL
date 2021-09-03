#include <ros/ros.h>
#include "pcl_utils/snapshot.h"

using namespace std;

int main (int argc, char** argv)
{
  ros::init(argc, argv, "get_pointcloud_client");
  ros::NodeHandle nh;
  ros::ServiceClient saveImage_client = nh.serviceClient<pcl_utils::snapshot>("snapshot");

  pcl_utils::snapshot snapshot_srv;

  snapshot_srv.request.call = 1;
  saveImage_client.call(snapshot_srv);

  return 0;
}