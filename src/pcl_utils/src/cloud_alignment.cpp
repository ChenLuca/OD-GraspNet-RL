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
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;

//=============
typedef pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;

pointcloud::Ptr Master_Cloud (new pointcloud);
pointcloud::Ptr Sub_Cloud (new pointcloud);
pointcloud::Ptr Top_Cloud (new pointcloud);

pointcloud::Ptr Master_Filter_Cloud (new pointcloud);
pointcloud::Ptr Sub_Filter_Cloud (new pointcloud);
pointcloud::Ptr Top_Filter_Cloud (new pointcloud);

pointcloud::Ptr Master_Rotate_Cloud (new pointcloud);
pointcloud::Ptr Sub_Rotate_Cloud (new pointcloud);
pointcloud::Ptr Top_Rotate_Cloud (new pointcloud);

pointcloud::Ptr Alignment_Cloud (new pointcloud);

sensor_msgs::PointCloud2 Alignment_Cloud_msg;
ros::Publisher pubAlignment_Cloud;

typedef boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> CloudPointRGBPtr;

struct ICPResult
{
  float icp_error;
  CloudPointRGBPtr cloud;
  Eigen::Matrix4d tf_matrix;
  float id;
};

float z_passthrough = 0.75;

//w, x, y, z
Eigen::Quaterniond quaterniond_master(0.50485725461, -0.489414097143, 0.488883808847, -0.51631929601);
Eigen::Quaterniond quaterniond_top(0.00677580204438, 0.999927007232, -0.00831166644454, -0.00556640961782);

Eigen::Matrix3d rotation_master_d = quaterniond_master.normalized().toRotationMatrix();
Eigen::Matrix3d rotation_top_d = quaterniond_top.normalized().toRotationMatrix();

Eigen::Matrix3f rotation_master = rotation_master_d.cast <float>();
Eigen::Matrix3f rotation_top = rotation_top_d.cast <float>();

//=============
void do_remove_outerpoint(pointcloud::Ptr &input_cloud, pointcloud::Ptr  &output_cloud)
{
  // pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
  // // build the filter
  // outrem.setInputCloud(input_cloud);
  // outrem.setRadiusSearch(0.05);
  // outrem.setMinNeighborsInRadius (100);
  // outrem.setKeepOrganized(true);
  // // apply filter
  // outrem.filter (*output_cloud);

  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud (input_cloud);
  sor.setMeanK (20);
  sor.setStddevMulThresh (1.0);
  sor.filter (*output_cloud);
}

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
  sor.setLeafSize (0.001, 0.001, 0.001);   //設定體素網格的大小
  sor.filter (*output_cloud);      //儲存濾波後的點雲
}

void do_Callback_PointCloud_Master(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{  
  // cout<<"Master " << cloud_msg->header.frame_id<<endl;
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *Master_Cloud);
  float pass_length = 0.25;
  do_Passthrough(Master_Cloud, Master_Filter_Cloud, "x", -pass_length, pass_length);
  do_Passthrough(Master_Filter_Cloud, Master_Filter_Cloud, "y", -pass_length, 0.13);
  do_Passthrough(Master_Filter_Cloud, Master_Filter_Cloud, "z", -1, 0.8);
  do_VoxelGrid(Master_Filter_Cloud, Master_Filter_Cloud);
  // do_remove_outerpoint(Master_Filter_Cloud, Master_Filter_Cloud);
  
}

void do_Callback_PointCloud_Sub(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{  
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *Sub_Cloud);
  // cout<<"Sub " <<  cloud_msg->header.frame_id<<endl;
  float pass_length = 0.5;

  do_Passthrough(Sub_Cloud, Sub_Filter_Cloud, "x", -pass_length, pass_length);
  do_Passthrough(Sub_Filter_Cloud, Sub_Filter_Cloud, "y", -pass_length, pass_length);
  do_Passthrough(Sub_Filter_Cloud, Sub_Filter_Cloud, "z", -1, 0.8);
  do_VoxelGrid(Sub_Filter_Cloud, Sub_Filter_Cloud);
  // do_remove_outerpoint(Sub_Filter_Cloud, Sub_Filter_Cloud);
}

void do_Callback_PointCloud_Top(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{  
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *Top_Cloud);
  // cout<<"Sub " <<  cloud_msg->header.frame_id<<endl;
  float pass_length = 0.25;

  do_Passthrough(Top_Cloud, Top_Filter_Cloud, "x", -pass_length, pass_length);
  do_Passthrough(Top_Filter_Cloud, Top_Filter_Cloud, "y", -pass_length-0.07, pass_length-0.075);
  do_Passthrough(Top_Filter_Cloud, Top_Filter_Cloud, "z", -1, z_passthrough);
  do_VoxelGrid(Top_Filter_Cloud, Top_Filter_Cloud);
  // do_remove_outerpoint(Top_Filter_Cloud, Top_Filter_Cloud);
}

Eigen::Matrix4d 
do_FPFH(pointcloud::Ptr  &target,
        pointcloud::Ptr  &source,
        pointcloud::Ptr  &output_cloud,
        double &FPFH_FitnessScore)
{
  pointcloud::Ptr cloud_tgt_o(new pointcloud);
  pointcloud::Ptr cloud_src_o(new pointcloud);

  cloud_tgt_o = target;
  cloud_src_o = source;

  // pcl::io::loadPCDFile<pcl::PointXYZRGB>(source, *cloud_src_o);

  //去除NAN点
	std::vector<int> indices_src; //保存去除的点的索引
	pcl::removeNaNFromPointCloud(*cloud_src_o, *cloud_src_o, indices_src);
	// std::cout << "remove *cloud_src_o nan" << endl;

	std::vector<int> indices_tgt;
	pcl::removeNaNFromPointCloud(*cloud_tgt_o, *cloud_tgt_o, indices_tgt);
	// std::cout << "remove *cloud_tgt_o nan" << endl;

  float downsample_rate = 0.002;
  //下采样滤波
	pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
	voxel_grid.setLeafSize(downsample_rate, downsample_rate, downsample_rate);
	voxel_grid.setInputCloud(cloud_src_o);
  pointcloud::Ptr cloud_src(new pointcloud);
	voxel_grid.filter(*cloud_src);
	std::cout << "down size *cloud_src_o from " << cloud_src_o->size() << "to" << cloud_src->size() << endl;

	pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid_2;
	voxel_grid_2.setLeafSize(downsample_rate, downsample_rate, downsample_rate);
	voxel_grid_2.setInputCloud(cloud_tgt_o);
  pointcloud::Ptr cloud_tgt(new pointcloud);
	voxel_grid_2.filter(*cloud_tgt);
	std::cout << "down size *cloud_tgt_o.pcd from " << cloud_tgt_o->size() << "to" << cloud_tgt->size() << endl;

  float ne_radius = 0.02;
	//计算表面法线
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_src;
	ne_src.setInputCloud(cloud_src);
	pcl::search::KdTree< pcl::PointXYZRGB>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZRGB>());
	ne_src.setSearchMethod(tree_src);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
	ne_src.setRadiusSearch(ne_radius);
	ne_src.compute(*cloud_src_normals);
	// std::cout << "compute *cloud_src_normals" << endl;

	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_tgt;
	ne_tgt.setInputCloud(cloud_tgt);
	pcl::search::KdTree< pcl::PointXYZRGB>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZRGB>());
	ne_tgt.setSearchMethod(tree_tgt);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
	//ne_tgt.setKSearch(20);
	ne_tgt.setRadiusSearch(ne_radius);
	ne_tgt.compute(*cloud_tgt_normals);
	// std::cout << "compute *cloud_tgt_normals" << endl;

  float fpfh_radius = 0.03; // must larger than ne_radius!

	//计算FPFH
	pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh_src;
	fpfh_src.setInputCloud(cloud_src);
	fpfh_src.setInputNormals(cloud_src_normals);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_src_fpfh(new pcl::search::KdTree<pcl::PointXYZRGB>);
	fpfh_src.setSearchMethod(tree_src_fpfh);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh_src.setRadiusSearch(fpfh_radius);
	fpfh_src.compute(*fpfhs_src);
	// std::cout << "compute *cloud_src fpfh" << endl;

	pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh_tgt;
	fpfh_tgt.setInputCloud(cloud_tgt);
	fpfh_tgt.setInputNormals(cloud_tgt_normals);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_tgt_fpfh(new pcl::search::KdTree<pcl::PointXYZRGB>);
	fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh_tgt.setRadiusSearch(fpfh_radius);
	fpfh_tgt.compute(*fpfhs_tgt);
	// std::cout << "compute *cloud_tgt fpfh" << endl;

	//SAC配准
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33> scia;
	scia.setInputSource(cloud_src);
	scia.setInputTarget(cloud_tgt);
	scia.setSourceFeatures(fpfhs_src);
	scia.setTargetFeatures(fpfhs_tgt);
	//scia.setMinSampleDistance(1);
	//scia.setNumberOfSamples(2);
	//scia.setCorrespondenceRandomness(20);
  pointcloud::Ptr sac_result(new pointcloud);

	scia.align(*sac_result);
	// std::cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
	Eigen::Matrix4d sac_trans;
	sac_trans = scia.getFinalTransformation().cast<double>();
	// std::cout << "sac_trans " << sac_trans << endl;
  FPFH_FitnessScore = scia.getFitnessScore();

  output_cloud = sac_result;
	// //pcl::io::savePCDFileASCII("bunny_transformed_sac.pcd", *sac_result);
  // pcl::toROSMsg(*output_cloud, FPFH_output);    //第一個引數是輸入，後面的是輸出
  // //Specify the frame that you want to publish
  // FPFH_output.header.frame_id = "camera_depth_optical_frame";
  // //釋出命令
  // pubFPFH.publish (FPFH_output);

  return sac_trans;
}



Eigen::Matrix4d  
do_ICP(pointcloud::Ptr &target, 
       pointcloud::Ptr &source,
       double &icp_FitnessScore)
{
  ICPResult tmp_result;

  //ICP
  int iterations = 50000*2;
  pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp; //创建ICP对象，用于ICP配准

  icp.setInputSource(source); //设置输入点云
  icp.setInputTarget(target); //设置目标点云（输入点云进行仿射变换，得到目标点云）
  // icp.setMaxCorrespondenceDistance(1);// Set the max correspondence distance (e.g., correspondences with higher distances will be ignored)
  icp.setMaximumIterations(iterations);    //criterion1: 设置最大迭代次数iterations=true
  icp.setTransformationEpsilon(1e-8);     //criterion2: transformation epsilon 

  icp.setEuclideanFitnessEpsilon(0.00001);   // Set the euclidean distance difference epsilon (criterion 3)    

  pointcloud::Ptr output_cloud(new pointcloud);
  
  output_cloud->clear();

  icp.align(*output_cloud);          //匹配后源点云
  
  // //icp.setMaximumIterations(1);  // 设置为1以便下次调用
  // std::cout << "Applied " << iterations << " ICP iteration(s)"<< std::endl;
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
  transformation_matrix = icp.getFinalTransformation().cast<double>();

  if (icp.hasConverged())//icp.hasConverged ()=1（true）输出变换矩阵的适合性评估
  {
      // std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
      transformation_matrix = icp.getFinalTransformation().cast<double>();
      tmp_result.icp_error = icp.getFitnessScore();
      tmp_result.cloud = output_cloud;
      tmp_result.tf_matrix = transformation_matrix;
      std::cout << "transformation_matrix "<<transformation_matrix<<endl;
      // icp_result.push_back(tmp_result);
      icp_FitnessScore = tmp_result.icp_error;
      cout << "ICP has converged" << endl;
  }
  else
  {
      PCL_ERROR("\nICP has not converged.\n");
  }  
  return transformation_matrix;
}

bool do_PointcloudProcess()
{
  Alignment_Cloud->clear();

  Eigen::Matrix4d FPFH_Transform, ICP_Transform;

  double ICP_FitnessScore, FPFH_FitnessScore;

  float master_factor = 1;
  float sub_factor = 1;
  float top_factor = 1;

  Eigen::Affine3f transform_master = Eigen::Affine3f::Identity();
  Eigen::Affine3f transform_top = Eigen::Affine3f::Identity();

  //Master flat
  transform_master.translation() << -0.440232458522 * master_factor, -0.63237217783 * master_factor, -0.1284713287062 * master_factor;
  
  transform_top.translation() <<  0.0325388687398 * top_factor, -0.702043973051 * top_factor, 0.523023032612 * top_factor;

  transform_master.rotate(rotation_master);

  transform_top.rotate(rotation_top);

  cout << "Master_Filter_Cloud->size() " << Master_Filter_Cloud->size() << endl;
  cout << "Top_Filter_Cloud->size() " << Top_Filter_Cloud->size() << endl;

  // if((Master_Filter_Cloud->size()!= 0) && (Top_Filter_Cloud->size()!= 0))
  if( (Top_Filter_Cloud->size()!= 0))

  {

      cout << "[cloud alignment] begin" << endl;

      // //solution 1: both transform to base [begin]
      // pcl::transformPointCloud(*Master_Filter_Cloud, *Master_Rotate_Cloud, transform_master);
      // pcl::transformPointCloud(*Top_Filter_Cloud, *Top_Rotate_Cloud, transform_top);
      // *Alignment_Cloud = *Master_Rotate_Cloud + *Top_Rotate_Cloud;

      // do_remove_outerpoint(Alignment_Cloud, Alignment_Cloud);

      // pcl::toROSMsg(*Alignment_Cloud, Alignment_Cloud_msg);
      // Alignment_Cloud_msg.header.frame_id = "base";
      // pubAlignment_Cloud.publish(Alignment_Cloud_msg);
      // //solution 1: both transform to base [end]

      //==================================================

      //solution 2 : master transform to top[begin]
      // pcl::transformPointCloud(*Master_Filter_Cloud, *Master_Rotate_Cloud, transform_master);
      // pcl::transformPointCloud(*Master_Rotate_Cloud, *Master_Rotate_Cloud, transform_top.inverse());
      // *Alignment_Cloud = *Master_Rotate_Cloud + *Top_Filter_Cloud;
      *Alignment_Cloud =  *Top_Filter_Cloud;
      
      do_remove_outerpoint(Alignment_Cloud, Alignment_Cloud);
      pcl::toROSMsg(*Alignment_Cloud, Alignment_Cloud_msg);
      Alignment_Cloud_msg.header.frame_id = "top_rgb_camera_link";
      pubAlignment_Cloud.publish(Alignment_Cloud_msg);
      //solution 2 : master transform to top[end]

      // cout << "Doing FPFH..." << endl;
      // FPFH_Transform = do_FPFH(Master_Rotate_Cloud, Sub_Rotate_Cloud, Alignment_Cloud, FPFH_FitnessScore);
      // cout << "Done FPFH..." << endl;
      // cout << "FPFH_FitnessScore " << FPFH_FitnessScore << endl;

      // cout << "Doing ICP..." << endl;
      // ICP_Transform = do_ICP(Master_Filter_Cloud, Alignment_Cloud, ICP_FitnessScore);
      // cout << "Done ICP..." << endl;
      
      cout << "[cloud alignment] end" << endl;

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
  // ros::Subscriber Sub_PointCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/sub/points2", 1, do_Callback_PointCloud_Sub);

  ros::Subscriber Top_PointCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/top/points2", 1, do_Callback_PointCloud_Top);


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