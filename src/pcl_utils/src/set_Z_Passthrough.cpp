#include <ros/ros.h>
#include "pcl_utils/setZPassthrough.h"

using namespace std;

int main (int argc, char** argv)
{
  ros::init(argc, argv, "set_Z_Passthrough_Clien");
  ros::NodeHandle nh;
  ros::ServiceClient set_Z_Passthrough_Clien = nh.serviceClient<pcl_utils::setZPassthrough>("set_z_passthrough");
  pcl_utils::setZPassthrough set_Z_Passthrough;
  float z_dist = atof(argv[1]);
  cout << "z_dist " << z_dist << endl;
  set_Z_Passthrough.request.z_dist = z_dist;
  set_Z_Passthrough_Clien.call(set_Z_Passthrough);

  return 0;
}