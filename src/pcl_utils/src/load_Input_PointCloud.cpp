#include <ros/ros.h>
#include "pcl_utils/loadPointCloud.h"

// #include <ros/ros.h>
// // PCL specific includes
// #include <sensor_msgs/PointCloud2.h>
// #include "geometry_msgs/PoseStamped.h"
// #include <visualization_msgs/Marker.h>
// #include <cv_bridge/cv_bridge.h>
// #include <image_transport/image_transport.h>
// #include <sensor_msgs/Image.h>
// #include <sensor_msgs/image_encodings.h>
// #include "std_msgs/Int64.h"
// #include "std_msgs/Float64.h"


// #include "pcl_utils/dl_grasp_result.h"
// #include "pcl_utils/AngleAxis_rotation_msg.h"
// #include "pcl_utils/coordinate_normal.h"
// #include "pcl_utils/RL_Env_msg.h"


// #include<opencv2/core/core.hpp>
// #include<opencv2/highgui/highgui.hpp>
// #include<opencv2/imgproc/imgproc.hpp>

// #include <typeinfo>

// #include "pcl_ros/filters/filter.h"
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/ModelCoefficients.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/passthrough.h>
// #include <pcl/common/transforms.h>
// #include <pcl/segmentation/region_growing_rgb.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/search/kdtree.h>
// #include <pcl/sample_consensus/method_types.h>
// #include <pcl/sample_consensus/model_types.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/registration/icp.h>
// #include <pcl/registration/icp_nl.h>
// #include <pcl/ModelCoefficients.h>
// #include <pcl/filters/project_inliers.h>
// #include <pcl/features/normal_3d.h>

// #include "pcl_utils/snapshot.h"
// #include "pcl_utils/setPointCloud.h"
// #include "pcl_utils/loadPointCloud.h"
// #include "pcl_utils/setZPassthrough.h"

// #include "pcl_utils/get_RL_Env.h"

// #include <iostream>
// #include <fstream>
// #include <algorithm>
// #include <string.h>
// #include <random>  

// //boost
// #include <boost/make_shared.hpp>
// #include <boost/thread/thread.hpp>
// #include <opencv2/calib3d.hpp>
// #include <std_msgs/Empty.h>

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