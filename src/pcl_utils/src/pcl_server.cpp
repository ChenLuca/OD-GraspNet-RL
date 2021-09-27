#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include "geometry_msgs/PoseStamped.h"
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <typeinfo>

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

#include "pcl_utils/snapshot.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include <random>  

//boost
#include <boost/make_shared.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

//=========Define ROS parameters=========
//pointcloud publish
ros::Publisher pubRotatePointClouds;

//image publish
image_transport::Publisher pubProjectDepthImage;
image_transport::Publisher pubProjectRGBImage;
  
//宣告的輸出的點雲的格式
sensor_msgs::PointCloud2 Filter_output;   
//==============================

//=========Define PCL parameters=========
//Origin Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

//Filter Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

//Rotate Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Rotate_output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//==============================

////=========global paremeters=========
//Project image size, maybe too large or small
int Mapping_width = 640, Mapping_high = 480;
cv::Mat Mapping_RGB_Image(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
cv::Mat Mapping_Depth_Image(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));
int take_picture_counter = 0;

//Intrinsics parameters
cv::Mat intrinsic_parameters(cv::Size(3, 3), cv::DataType<float>::type); 
cv::Mat distortion_coefficient(cv::Size(5, 1), cv::DataType<float>::type);

//Intrinsics parameters from depth camera
double cx = 325.506, cy = 332.234, fx = 503.566, fy = 503.628;

bool SHOW_CV_WINDOWS = false;
bool ROTATE_POINTCLOUD =false;
//==============================

////=========random=========
std::random_device rd;
std::default_random_engine generator( rd() );
std::uniform_real_distribution<float> unif(0.0, 1.0);
//==============================

void do_ViewpointTrasform(Eigen::Matrix4d &viewpoint_transform, float *translation, float *rotation)
{
  viewpoint_transform(0,0) = cos(rotation[2])*cos(rotation[1]);
  viewpoint_transform(0,1) = (cos(rotation[2])*sin(rotation[1])*sin(rotation[0])) - (sin(rotation[2])*cos(rotation[0]));
  viewpoint_transform(0,2) = (cos(rotation[2])*sin(rotation[1])*cos(rotation[0])) + (sin(rotation[2])*sin(rotation[0]));
  viewpoint_transform(0,3) = translation[0];
  viewpoint_transform(1,0) = sin(rotation[2])*cos(rotation[1]);
  viewpoint_transform(1,1) = (sin(rotation[2])*sin(rotation[1])*sin(rotation[0])) + (cos(rotation[2])*cos(rotation[0]));
  viewpoint_transform(1,2) = (sin(rotation[2])*sin(rotation[1])*cos(rotation[0])) - (cos(rotation[2])*sin(rotation[0]));
  viewpoint_transform(1,3) = translation[1];
  viewpoint_transform(2,0) = -sin(rotation[1]);
  viewpoint_transform(2,1) = cos(rotation[1])*sin(rotation[0]);
  viewpoint_transform(2,2) = cos(rotation[1])*cos(rotation[0]);
  viewpoint_transform(2,3) = translation[2];
}

//My methods for mapping pointcloud data to 2d image, will not be used in future.
void do_MappingPointCloud2image(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud)
{
  Eigen::Matrix3f RGB_Intrinsic_Matrix = Eigen::Matrix3f::Identity();
  Eigen::Matrix<float, 3, 4> RGB_Extrinsic_Matrix;
  Eigen::Matrix<float, 3, 4> RGB_Matrix; // A[R|t]
  Eigen::Matrix<float, 4, 1> PointCloud_Matrix; // [x, y, z, 1]
  Eigen::Matrix<float, 3, 1> Single_RGB_point; // [x, y, 1]

  double cx, cy, fx, fy;
  cx = 639.739;
  cy = 366.687;
  fx = 606.013;
  fy = 605.896;

  RGB_Intrinsic_Matrix (0, 0) = fx;
  RGB_Intrinsic_Matrix (0, 1) = 0.0;
  RGB_Intrinsic_Matrix (0, 2) = cx;
  RGB_Intrinsic_Matrix (1, 0) = 0.0;
  RGB_Intrinsic_Matrix (1, 1) = fy;
  RGB_Intrinsic_Matrix (1, 2) = cy;
  RGB_Intrinsic_Matrix (2, 0) = 0.0;
  RGB_Intrinsic_Matrix (2, 1) = 0.0;
  RGB_Intrinsic_Matrix (2, 2) = 1.0;

  RGB_Extrinsic_Matrix (0, 0) =  0.999977;
  RGB_Extrinsic_Matrix (0, 1) = 0.00682134;
  RGB_Extrinsic_Matrix (0, 2) = 0.0003813;
  RGB_Extrinsic_Matrix (0, 3) = -32.002/1000;
  RGB_Extrinsic_Matrix (1, 0) = -0.00682599;
  RGB_Extrinsic_Matrix (1, 1) = 0.995201;
  RGB_Extrinsic_Matrix (1, 2) = 0.097608;
  RGB_Extrinsic_Matrix (1, 3) = -1.96833/1000;
  RGB_Extrinsic_Matrix (2, 0) = 0.00028632;
  RGB_Extrinsic_Matrix (2, 1) = -0.0976088;
  RGB_Extrinsic_Matrix (2, 2) = 0.99522;
  RGB_Extrinsic_Matrix (2, 3) = 4.01666/1000;

  RGB_Matrix = RGB_Intrinsic_Matrix * RGB_Extrinsic_Matrix;

  int width = 1800, high = 1800;

  cv::Mat Mapping_RGB_Image(high, width, CV_32FC3, cv::Scalar(0, 0, 0));
  
  if (input_cloud->size()!= 0)
  {
    for (int i = 0 ; i < input_cloud->size(); i++)
    {
      PointCloud_Matrix (0) =  input_cloud->points[i].x;
      PointCloud_Matrix (1) =  input_cloud->points[i].y;
      PointCloud_Matrix (2) =  input_cloud->points[i].z;
      PointCloud_Matrix (3) =  1.0;

      float b = (int)input_cloud->points[i].b/255.0;
      float g = (int)input_cloud->points[i].g/255.0;
      float r = (int)input_cloud->points[i].r/255.0;

      Single_RGB_point = RGB_Matrix * PointCloud_Matrix;

      //   Point(x, y) need to Divide by z! ensure that -> [x, y, "1"]
      //   e.g.
      //   point2d = matrix(Intrinsic * Extrinsic) * point3d;
      //   point2d = point2d / point2d.z;
      int Single_RGB_point_x = (int)(cx + Single_RGB_point(0)/Single_RGB_point(2));
      int Single_RGB_point_y = (int)(cy + Single_RGB_point(1)/Single_RGB_point(2));
      
      if (Single_RGB_point_x < 0 || Single_RGB_point_y < 0 || Single_RGB_point_x > width || Single_RGB_point_y > high )
      {
        cout << "Single_RGB_point_x "<<Single_RGB_point_x<<endl;
        cout << "Single_RGB_point_y "<<Single_RGB_point_y<<endl;
        cout << "break!!" <<endl;
        break;
      }
      Mapping_RGB_Image.at<cv::Vec3f>(Single_RGB_point_y, Single_RGB_point_x)[0] = b;
      Mapping_RGB_Image.at<cv::Vec3f>(Single_RGB_point_y, Single_RGB_point_x)[1] = g;
      Mapping_RGB_Image.at<cv::Vec3f>(Single_RGB_point_y, Single_RGB_point_x)[2] = r;
    }
    // cv::resize(Mapping_RGB_Image, Mapping_RGB_Image, cv::Size(1280, 720), cv::INTER_AREA);
    cv::namedWindow("Image window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image window", Mapping_RGB_Image);
    cv::waitKey(1);
  }
}

//for DoPerspectiveProjection()
void CloudToVector(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pt_cloud, std::vector<cv::Point3f> & pt_vect)
{
	for (int n = 0; n < pt_cloud->size(); ++n)
	{
		pcl::PointXYZRGB point = pt_cloud->points[n];
		cv::Point3f pt(point.x, point.y, point.z);
		pt_vect.push_back(pt);
	}
}

//for DoPerspectiveProjection()
void Matrix4dToRodriguesTranslation(Eigen::Matrix4d & matrix, cv::Vec3d & rvec, cv::Vec3d & tvec)
{
	rvec = { 0.0, 0.0, 0.0 };
	tvec = { 0.0, 0.0, 0.0 };

	//opencv
	cv::Mat m33 = cv::Mat::eye(3, 3, CV_64FC1);//CV_32FC1
	m33.at<double>(0) = matrix(0, 0);
	m33.at<double>(1) = matrix(0, 1);
	m33.at<double>(2) = matrix(0, 2);
	m33.at<double>(3) = matrix(1, 0);
	m33.at<double>(4) = matrix(1, 1);
	m33.at<double>(5) = matrix(1, 2);
	m33.at<double>(6) = matrix(2, 0);
	m33.at<double>(7) = matrix(2, 1);
	m33.at<double>(8) = matrix(2, 2);
	cv::Rodrigues(m33, rvec);

	for (int i = 0; i < 3; ++i)
		tvec[i] = matrix(i, 3);
}

//OpenCV methods for mapping pointcloud data to 2d image
void do_PerspectiveProjection(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, cv::Mat &Mapping_RGB_Image, cv::Mat &Mapping_Depth_Image, float *viewpoint_translation, float *viewpoint_Rotation)
{
  //Extrinsics parameters
  Eigen::Matrix4d cam1_H_world = Eigen::Matrix4d::Identity();
  
  intrinsic_parameters.at<float> (0, 0) = fx;
  intrinsic_parameters.at<float> (0, 1) = 0.0;
  intrinsic_parameters.at<float> (0, 2) = cx;
  intrinsic_parameters.at<float> (1, 0) = 0.0;
  intrinsic_parameters.at<float> (1, 1) = fy;
  intrinsic_parameters.at<float> (1, 2) = cy;
  intrinsic_parameters.at<float> (2, 0) = 0.0;
  intrinsic_parameters.at<float> (2, 1) = 0.0;
  intrinsic_parameters.at<float> (2, 2) = 1.0;

  //k1,k2,p1,p2,k3
  distortion_coefficient.at<float> (0) = 0.0;
  distortion_coefficient.at<float> (1) = 0.0;
  distortion_coefficient.at<float> (2) = 0.0;
  distortion_coefficient.at<float> (3) = 0.0;
  distortion_coefficient.at<float> (4) = 0.0;

  // //Extrinsics parameters from depth camera
  // cam1_H_world (0, 0) = 1.0;
  // cam1_H_world (0, 1) = 0.0;
  // cam1_H_world (0, 2) = 0.0;
  // cam1_H_world (0, 3) = 0.0;
  // cam1_H_world (1, 0) = 0.0;
  // cam1_H_world (1, 1) = 1.0;
  // cam1_H_world (1, 2) = 0.0;
  // cam1_H_world (1, 3) = 0.0;
  // cam1_H_world (2, 0) = 0.0;
  // cam1_H_world (2, 1) = 0.0;
  // cam1_H_world (2, 2) = 1.0;
  // cam1_H_world (2, 3) = 0.0;

  // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  Eigen::Matrix4d viewpoint_transform = Eigen::Matrix4d::Identity();

  do_ViewpointTrasform(viewpoint_transform, viewpoint_translation, viewpoint_Rotation);
  
  cam1_H_world = viewpoint_transform;

  // Read 3D points: cloud-> vector
	std::vector<cv::Point3f> cloudPoints;
	CloudToVector(input_cloud, cloudPoints);

	cv::Vec3d cam1_H_world_rvec, cam1_H_world_tvec;
	Matrix4dToRodriguesTranslation(cam1_H_world, cam1_H_world_rvec, cam1_H_world_tvec);

	// Perspective Projection of Cloud Points to Image Plane
	std::vector<cv::Point2f> imagePoints;
	cv::projectPoints(cloudPoints, cam1_H_world_rvec, cam1_H_world_tvec, intrinsic_parameters, distortion_coefficient, imagePoints);

  if (input_cloud->size()!= 0)
  {
    float depth_interval = 0.013176470588235293;
    float z = 0.0;
    int idxX = 0, idxY = 0;

    for (unsigned int i = 0; i < imagePoints.size(); ++i) 
    {
      idxX = round(imagePoints[i].x);
      idxY = round(imagePoints[i].y);

      if (idxX < 0 || idxY < 0 || idxX >= Mapping_width || idxY >= Mapping_high)
      {
        continue;
      }

      // float b = (int)input_cloud->points[i].b;
      // float g = (int)input_cloud->points[i].g;
      // float r = (int)input_cloud->points[i].r;
      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[0] = (int)input_cloud->points[i].b;
      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[1] = (int)input_cloud->points[i].g;
      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[2] = (int)input_cloud->points[i].r;

      z = input_cloud->points[i].z;

      if (z > 0.5 && z < 3.86)
      {
        z = (z-0.5) / depth_interval;
        Mapping_Depth_Image.at<uchar>(idxY, idxX) = round(z);
      }
    }
  }
  // return Mapping_RGB_Image;
}


Eigen::Vector4f do_ComputeLocation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud)
{
  Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*input_cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*input_cloud, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //校正主方向间垂直
	eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
	eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
 
	// std::cout << "特徵值va(3x1):\n" << eigenValuesPCA << std::endl;
	// std::cout << "特徵向量ve(3x3):\n" << eigenVectorsPCA << std::endl;
	// std::cout << "質心點(4x1):\n" << pcaCentroid << std::endl;
  return pcaCentroid;
}

void do_Rotate(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud,
               Eigen::Vector4f pca_location,
               float *Rotate_angle)
{
  
  Eigen::Affine3f transform_trans = Eigen::Affine3f::Identity();
  Eigen::Affine3f transform_rotate = Eigen::Affine3f::Identity();

  // Define a translation .
  transform_trans.translation() << -1*pca_location[0], -1*pca_location[1], -1*pca_location[2];

  // The same rotation matrix as before; theta radians around Z axis
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[2], Eigen::Vector3f::UnitZ()));
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[1], Eigen::Vector3f::UnitY()));
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[0], Eigen::Vector3f::UnitX()));

  // Apply an affine transform defined by an Eigen Transform.
  pcl::transformPointCloud (*input_cloud, *output_cloud, transform_trans);
  pcl::transformPointCloud (*output_cloud, *output_cloud, transform_rotate);
  transform_trans.translation() << pca_location[0], pca_location[1], pca_location[2];
  pcl::transformPointCloud (*output_cloud, *output_cloud, transform_trans);

}

void do_Passthrough(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud,
                    std::string dim, float min, float max)
{
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (input_cloud);
  pass.setFilterFieldName (dim);
  pass.setFilterLimits (min, max);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*output_cloud);
}

void do_VoxelGrid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud)
{
  // 進行一個濾波處理
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;   //例項化濾波
  sor.setInputCloud (input_cloud);     //設定輸入的濾波
  sor.setLeafSize (0.005, 0.005, 0.005);   //設定體素網格的大小
  sor.filter (*output_cloud);      //儲存濾波後的點雲
}

void do_Callback_PointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{  
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *cloud);

  do_Passthrough(cloud, filter_cloud, "x", -0.28, 0.35);
  do_Passthrough(filter_cloud, filter_cloud, "y", -1, 0.08);
  do_Passthrough(filter_cloud, filter_cloud, "z", -1, 0.85);
  do_VoxelGrid(filter_cloud, filter_cloud);

  if(ROTATE_POINTCLOUD)
  {
    float *Rotate_angle = new float[3];
    Rotate_angle[0] = 0;  //X axis
    Rotate_angle[1] = 0;  //Y axis
    Rotate_angle[2] = 0;  //Z axis

    //rotate point Cloud
    do_Rotate(filter_cloud, 
              Rotate_output_cloud, 
              do_ComputeLocation(filter_cloud), 
              Rotate_angle);

    // PCL to ROS, 第一個引數是輸入，後面的是輸出
    pcl::toROSMsg(*Rotate_output_cloud, Filter_output);

    //Specify the frame that you want to publish
    Filter_output.header.frame_id = "depth_camera_link";

    //釋出命令
    pubRotatePointClouds.publish (Filter_output);
  }

  //reset values in the images
  Mapping_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
  Mapping_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));

  //get random number
  float random_rotation = unif(generator);
  // std::cout << "random_rotation = " << random_rotation << std::endl;

  //viewpoint transform
  float *viewpoint_translation = new float[3];
  float *viewpoint_Rotation = new float[3];

  viewpoint_translation[0] = 0.0;
  viewpoint_translation[1] = 0.0;
  viewpoint_translation[2] = 0.0;
  viewpoint_Rotation[0] = 0.0;
  viewpoint_Rotation[1] = 0.0;
  viewpoint_Rotation[2] = 0.0;

  do_PerspectiveProjection(filter_cloud, Mapping_RGB_Image, Mapping_Depth_Image, viewpoint_translation, viewpoint_Rotation);

  //do dilate for sparse image result
  cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));  
  cv::dilate(Mapping_RGB_Image, Mapping_RGB_Image, element);
  cv::dilate(Mapping_Depth_Image, Mapping_Depth_Image, element);

  if(SHOW_CV_WINDOWS)
  {
    cv::namedWindow("Depth Image window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Depth Image window", Mapping_Depth_Image);
    cv::namedWindow("RGB Image window", cv::WINDOW_AUTOSIZE);
    cv::imshow("RGB Image window", Mapping_RGB_Image);
    cv::waitKey(1);
  }

  //====publish image to ros topic
  sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", Mapping_RGB_Image).toImageMsg();
  ros::Time rgb_begin = ros::Time::now();
  rgb_msg->header.stamp = rgb_begin;
  pubProjectRGBImage.publish(rgb_msg);

  sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", Mapping_Depth_Image).toImageMsg();
  ros::Time depth_begin = ros::Time::now();
  depth_msg->header.stamp = depth_begin;
  pubProjectDepthImage.publish(depth_msg);

  ROS_INFO("Done do_Callback_PointCloud");
}

string SaveImage_Counter_Wrapper(int num, int object_number)
{
  string Complement = "";

  if (num < 10)
  {
    Complement = to_string(object_number) + "0";
  }
  else
  {
    Complement = to_string(object_number);
  }

  return (Complement + to_string(num));
}

bool do_SaveImage(pcl_utils::snapshot::Request &req, pcl_utils::snapshot::Response &res)
{
  string Save_Data_path = "/home/ur5/datasets/my_grasp_dataset/";
  string Name_pcd = "pcd";
  string Name_RGB_Image_root = "r.png";
  string Name_Depth_Image_root = "d.tiff";

  //save RGB image as .png format
  cv::imwrite(Save_Data_path + Name_pcd + SaveImage_Counter_Wrapper(take_picture_counter, req.call) + Name_RGB_Image_root, Mapping_RGB_Image);

  //save depth image as .tiff format
  cv::imwrite(Save_Data_path + Name_pcd + SaveImage_Counter_Wrapper(take_picture_counter, req.call) + Name_Depth_Image_root, Mapping_Depth_Image);

  take_picture_counter++;

  res.back = take_picture_counter;

  ROS_INFO("Done SaveImage!");

  return true;
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "push_to_cloud_stitched");

  ros::NodeHandle nh;
  
  // Create ROS publisher for projected image
  image_transport::ImageTransport it(nh);
  pubProjectRGBImage = it.advertise("/projected_image/rgb", 1);
  pubProjectDepthImage = it.advertise("/projected_image/depth", 1);

  // Create ROS preccess pointcloud publisher for the rotate point cloud
  pubRotatePointClouds = nh.advertise<sensor_msgs::PointCloud2> ("/Rotate_PointClouds", 30);
  
  // Create ROS subscriber for the input point cloud
  //azure kinect dk
  ros::Subscriber subSaveCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/points2", 1, do_Callback_PointCloud);

  // Create ROS Service for the input point cloud
  ros::ServiceServer saveImage_service = nh.advertiseService("snapshot", do_SaveImage);


  ros::spin();
}
