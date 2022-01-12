#include <ros/ros.h>
#include <iostream>
#include "pcl_ros/filters/filter.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh_omp.h> //包含fpfh加速計算的omp(多核平行計算)
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h> //特徵的錯誤對應關係去除
#include <pcl/registration/correspondence_rejection_sample_consensus.h> //隨機取樣一致性去除
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>

using namespace std;

//=============
typedef pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

pointcloud::Ptr Master_Cloud (new pointcloud);
pointcloud::Ptr Sub_Cloud (new pointcloud);
pointcloud::Ptr Master_Filter_Cloud (new pointcloud);
pointcloud::Ptr Sub_Filter_Cloud (new pointcloud);
pointcloud::Ptr Alignment_Cloud (new pointcloud);

sensor_msgs::PointCloud2 Alignment_Cloud_msg;
ros::Publisher pubAlignment_Cloud;

float z_passthrough = 1.5;
//=============

void do_Passthrough(pointcloud::Ptr &input_cloud, 
                    pointcloud::Ptr &output_cloud,
                    std::string dim, float min, float max)
{
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (input_cloud);
  pass.setFilterFieldName (dim);
  pass.setFilterLimits (min, max);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*output_cloud);
}

void do_VoxelGrid(pointcloud::Ptr &input_cloud, 
                    pointcloud::Ptr &output_cloud)
{
  // 進行一個濾波處理
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;   //例項化濾波
  sor.setInputCloud (input_cloud);     //設定輸入的濾波
  sor.setLeafSize (0.01, 0.01, 0.01);   //設定體素網格的大小
  sor.filter (*output_cloud);      //儲存濾波後的點雲
}

void do_Callback_PointCloud_Master(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{  
  // cout<<"Master " << cloud_msg->header.frame_id<<endl;
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *Master_Cloud);

  do_Passthrough(Master_Cloud, Master_Filter_Cloud, "x", -1.0, 1.0);
  do_Passthrough(Master_Filter_Cloud, Master_Filter_Cloud, "y", -1.0, 1.0);
  do_Passthrough(Master_Filter_Cloud, Master_Filter_Cloud, "z", -1, z_passthrough);
  do_VoxelGrid(Master_Filter_Cloud, Master_Filter_Cloud);
  
}

void do_Callback_PointCloud_Sub(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{  
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *Sub_Cloud);
  // cout<<"Sub " <<  cloud_msg->header.frame_id<<endl;

  do_Passthrough(Sub_Cloud, Sub_Filter_Cloud, "x", -1.0, 1.0);
  do_Passthrough(Sub_Filter_Cloud, Sub_Filter_Cloud, "y", -1.0, 1.0);
  do_Passthrough(Sub_Filter_Cloud, Sub_Filter_Cloud, "z", -1, z_passthrough);
  do_VoxelGrid(Sub_Filter_Cloud, Sub_Filter_Cloud);
  
}

fpfhFeature::Ptr compute_fpfh_feature(pointcloud::Ptr input_cloud, pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree)
{
  //法向量
  pointnormal::Ptr point_normal (new pointnormal);
  pcl::NormalEstimation<pcl::PointXYZRGB,pcl::Normal> est_normal;
  est_normal.setInputCloud(input_cloud);
  est_normal.setSearchMethod(tree);
  est_normal.setKSearch(10);
  // est_normal.setRadiusSearch(0.03);
  est_normal.compute(*point_normal);

  //fpfh 估計
  fpfhFeature::Ptr fpfh (new fpfhFeature);

  //pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> est_target_fpfh;
  pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> est_fpfh;

  est_fpfh.setNumberOfThreads(8); //指定4核計算
  // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree4 (new pcl::search::KdTree<pcl::PointXYZ> ());

  est_fpfh.setInputCloud(input_cloud);
  est_fpfh.setInputNormals(point_normal);
  est_fpfh.setSearchMethod(tree);
  est_fpfh.setKSearch(10);
  est_fpfh.compute(*fpfh);
  return fpfh;
}

void do_FPFH(pointcloud::Ptr source, pointcloud::Ptr target, pointcloud::Ptr align)
{
  clock_t start, end, time;
  start  = clock();

  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr source_tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr target_tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

  fpfhFeature::Ptr source_fpfh =  compute_fpfh_feature(source, source_tree);
  fpfhFeature::Ptr target_fpfh =  compute_fpfh_feature(target, target_tree);

  //對齊(佔用了大部分執行時間)
  pcl::SampleConsensusPrerejective<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33> scp;
  scp.setInputSource(source);
  scp.setSourceFeatures(source_fpfh);
  scp.setInputTarget(target);
  scp.setTargetFeatures(target_fpfh);

  scp.setNumberOfSamples(3);  //設定每次迭代計算中使用的樣本數量（可省）,可節省時間
  scp.setCorrespondenceRandomness(5); //設定計算協方差時選擇多少近鄰點，該值越大，協防差越精確，但是計算效率越低.(可省)
  scp.setMaximumIterations(50000);
  scp.setSimilarityThreshold(0.9f);
  scp.setMaxCorrespondenceDistance(2.5f * 0.005);
  scp.setInlierFraction(0.25f);
  scp.align(*align); 

  if (!scp.hasConverged())
  {
    cout << "Alignment failed!" << endl;
  }
  else
  {
    Eigen::Matrix4f transformation = scp.getFinalTransformation();
    cout << "R" << endl;
    cout << transformation(0, 0) << ", " << transformation(0, 1) << ", " << transformation(0, 2) << endl;
    cout << transformation(1, 0) << ", " << transformation(1, 1) << ", " << transformation(1, 2) << endl;
    cout << transformation(2, 0) << ", " << transformation(2, 1) << ", " << transformation(2, 2) << endl;
    cout << "t" << endl;
    cout << transformation(0, 3) << ", " << transformation(1, 3) << ", " << transformation(2, 3) << endl;
  }

  end = clock();
  cout <<"calculate time is: "<< float (end-start)/CLOCKS_PER_SEC<<endl;
}

bool do_PointcloudProcess()
{
  Alignment_Cloud->clear();

  // cout << "Master_Filter_Cloud->size() " << Master_Filter_Cloud->size() << endl; 
  // cout << "Sub_Filter_Cloud->size() " << Sub_Filter_Cloud->size() << endl; 

  if((Master_Filter_Cloud->size()!= 0) && (Sub_Filter_Cloud->size()!= 0))
  {
      cout << "Doing FPFH..." << endl;
      do_FPFH(Master_Filter_Cloud, Sub_Filter_Cloud, Alignment_Cloud);
      cout << "Done FPFH..." << endl;

      pcl::toROSMsg(*Alignment_Cloud, Alignment_Cloud_msg);
      Alignment_Cloud_msg.header.frame_id = "master_rgb_camera_link";
      pubAlignment_Cloud.publish(Alignment_Cloud_msg);
  }
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pointcloud_alignment");

  // Initialize NodeHandle
  ros::NodeHandle nh;

  // Create ROS subscriber for the input master point cloud (azure kinect dk)
  ros::Subscriber Master_PointCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/master/points2", 1, do_Callback_PointCloud_Master);

  // Create ROS subscriber for the sub input point cloud (azure kinect dk)
  ros::Subscriber Sub_PointCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/sub/points2", 1, do_Callback_PointCloud_Sub);

  // Create ROS pointcloud publisher for the point cloud of Alignment_Cloud
  pubAlignment_Cloud = nh.advertise<sensor_msgs::PointCloud2> ("/Alignment_Cloud", 30);

  ros::Rate loop_rate(100);

  while(ros::ok())
  {
    do_PointcloudProcess();
    ros::spinOnce();
    loop_rate.sleep();
  }
  
  return 0;
}