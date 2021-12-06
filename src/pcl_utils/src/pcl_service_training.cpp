#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include "geometry_msgs/PoseStamped.h"
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include "std_msgs/Int64.h"
#include "std_msgs/Float64.h"


#include "pcl_utils/grcnn_result.h"
#include "pcl_utils/AngleAxis_rotation_msg.h"
#include "pcl_utils/coordinate_normal.h"


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
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/normal_3d.h>

#include "pcl_utils/snapshot.h"

#include "pcl_utils/setPointCloud.h"
#include "pcl_utils/loadPointCloud.h"
#include "pcl_utils/setZPassthrough.h"


#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include <random>  

//boost
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>
#include <opencv2/calib3d.hpp>
#include <std_msgs/Empty.h>

using namespace std;

//=========Define ROS parameters=========
//pointcloud publish
ros::Publisher pubRotatePointClouds, pubGrabPointClouds, pubNumGrabPoint, pubNumFingerGrabPoint, pubGrabPointCloudsLeft, pubGrabPointCloudsRight, pubGrabPointCloudsLeftNormal, pubGrabPointCloudsRightNormal, 
               pubGrabPointCloudsNormal,
               pubAngleAxisOpen, pubAngleAxisApproach, pubAngleAxisNormal, 
               pubProjectNormalVectorPlaneCloud, pubProjectApproachVectorPlaneCloud, pubProjectOpenVectorPlaneCloud,
               pubRetransformProjectNormalVectorPlaneCloud, pubRetransformProjectApproachVectorPlaneCloud, pubRetransformProjectOpenVectorPlaneCloud,
               pubRightFingerPoint,
               pub_pose_left, pub_pose_right, pub_pose_grab,
               pubLeftLikelihood, pubRightLikelihood, pubApproachLikelihood, pubNormalLikelihood,
               pubNormaldepthNonZero, pubOpendepthNonZero,
               pubNowCloud;

//image publish
image_transport::Publisher pubProjectDepthImage;
image_transport::Publisher pubProjectRGBImage;
image_transport::Publisher pubProject_Grab_Approach_RGB_Image;
image_transport::Publisher pubProject_Grab_Approach_Depth_Image;
image_transport::Publisher pubProject_Grab_Normal_RGB_Image;
image_transport::Publisher pubProject_Grab_Normal_Depth_Image;
image_transport::Publisher pubProject_Grab_Open_RGB_Image;
image_transport::Publisher pubProject_Grab_Open_Depth_Image;

//宣告的輸出的點雲的格式
sensor_msgs::PointCloud2 Filter_output, grab_output, grab_output_left, grab_output_right,
                         project_normal_vector_plane_output, 
                         project_approach_vector_plane_output, 
                         project_open_vector_plane_output,
                         retransform_project_normal_vector_plane_output, 
                         retransform_project_approach_vector_plane_output, 
                         retransform_project_open_vector_plane_output,
                         now_cloud_output;


visualization_msgs::Marker open_arrow, normal_arrow, approach_arrow, right_finger_point;

//==============================

//=========Define PCL parameters=========
//Origin Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

//Filter Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud process right now
pcl::PointCloud<pcl::PointXYZRGB>::Ptr now_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

//Rotated Pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Rotate_output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of gripped area
pcl::PointCloud<pcl::PointXYZRGB>::Ptr grab_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of left gripped area
pcl::PointCloud<pcl::PointXYZRGB>::Ptr grab_cloud_left(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of right gripped area
pcl::PointCloud<pcl::PointXYZRGB>::Ptr grab_cloud_right(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of projected normal vector plane cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr project_normal_vector_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of projected approach vector plane cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr project_approach_vector_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of projected open vector plane cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr project_open_vector_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of retransform projected normal vector plane cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr retransform_normal_vector_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of retransform projected approach vector plane cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr retransform_approach_vector_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//Pointcloud of retransform projected open vector plane cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr retransform_open_vector_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);


struct Point_with_Pixel
{
  cv::Point3f point;
  cv::Point2f pixel; //pixel rounded
};

struct AxisQuaterniond
{
  Eigen::Quaterniond open_q;
  Eigen::Quaterniond approach_q;
  Eigen::Quaterniond normal_q;
};

struct oan_vector
{
  Eigen::Vector4d open_vector;
  Eigen::Vector4d approach_vector;
  Eigen::Vector4d normal_vector;
};
//==============================

////=========global paremeters=========

bool ROTATE_POINTCLOUD = false;
bool Set_Input_PointCloud = false;
//Snapshot service
int take_picture_counter = 0;

//Project image size, maybe too large or small
int Mapping_width = 640, Mapping_high = 480;
cv::Mat Mapping_RGB_Image(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
cv::Mat Mapping_Depth_Image(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));

cv::Mat Grab_Cloud_Approach_RGB_Image(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
cv::Mat Grab_Cloud_Approach_Depth_Image(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));
cv::Mat Grab_Cloud_Normal_RGB_Image(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
cv::Mat Grab_Cloud_Normal_Depth_Image(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));
cv::Mat Grab_Cloud_Open_RGB_Image(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
cv::Mat Grab_Cloud_Open_Depth_Image(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));

//Intrinsics parameters
cv::Mat intrinsic_parameters(cv::Size(3, 3), cv::DataType<float>::type); 
cv::Mat distortion_coefficient(cv::Size(5, 1), cv::DataType<float>::type);

//Intrinsics parameters from depth camera
double cx = 325.506, cy = 332.234, fx = 503.566, fy = 503.628;

//Grab pointcloud Rotation
float Angle_axis_rotation_open = 0.0, Angle_axis_rotation_approach = 0.0, Angle_axis_rotation_normal = 0.0;

//GRCNN input
pcl_utils::grcnn_result grcnn_input;

//viewpoint transform
float *viewpoint_translation = new float[3];
float *viewpoint_rotation = new float[3];
float *grasp_3D = new float[3];
float *Grab_Cloud_viewpoint_Translation = new float[3];
float *Grab_Cloud_viewpoint_Rotation = new float[3];

int NumberOfLocalPCD = 20;
int nowLocalPCD = 0;
float max_execution_time = 0;
float z_passthrough = 0.8;

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
    // cv::namedWindow("Image window", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Image window", Mapping_RGB_Image);
    // cv::waitKey(1);
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
void do_PerspectiveProjection(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, cv::Mat &Mapping_RGB_Image, 
                              cv::Mat &Mapping_Depth_Image, float *viewpoint_Translation, float *viewpoint_Rotation,
                              std::vector<Point_with_Pixel> &PwPs, float intrinsic_fx, float intrinsic_fy, float intrinsic_cx, float intrinsic_cy)
{
  //Extrinsics parameters
  Eigen::Matrix4d cam1_H_world = Eigen::Matrix4d::Identity();
  
  intrinsic_parameters.at<float> (0, 0) = intrinsic_fx;
  intrinsic_parameters.at<float> (0, 1) = 0.0;
  intrinsic_parameters.at<float> (0, 2) = intrinsic_cx;
  intrinsic_parameters.at<float> (1, 0) = 0.0;
  intrinsic_parameters.at<float> (1, 1) = intrinsic_fy;
  intrinsic_parameters.at<float> (1, 2) = intrinsic_cy;
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

  do_ViewpointTrasform(viewpoint_transform, viewpoint_Translation, viewpoint_Rotation);
  
  cam1_H_world = viewpoint_transform;

  // Read 3D points: cloud-> vector
	std::vector<cv::Point3f> cloudPoints;
	CloudToVector(input_cloud, cloudPoints);
  
	cv::Vec3d cam1_H_world_rvec, cam1_H_world_tvec;
	Matrix4dToRodriguesTranslation(cam1_H_world, cam1_H_world_rvec, cam1_H_world_tvec);

	// Perspective Projection of Cloud Points to Image Plane
	std::vector<cv::Point2f> imagePoints;
	cv::projectPoints(cloudPoints, cam1_H_world_rvec, cam1_H_world_tvec, intrinsic_parameters, distortion_coefficient, imagePoints);

  Point_with_Pixel PwP;

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

      PwP.point = cloudPoints[i];
      PwP.pixel.x = idxX;
      PwP.pixel.y = idxY;
      PwPs.push_back(PwP);

      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[0] = (int)input_cloud->points[i].b;
      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[1] = (int)input_cloud->points[i].g;
      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[2] = (int)input_cloud->points[i].r;

      z = input_cloud->points[i].z;
      // if (z > 0.0 && z < 3.86)
      // {
      //   z = (z) / depth_interval;
      //   Mapping_Depth_Image.at<uchar>(idxY, idxX) = round(z);
      // }
      if (z > 0.0 && z < 1.0)
      {
        z = z/1.0*255;
        Mapping_Depth_Image.at<uchar>(idxY, idxX) = round(z);
      }
    }
  }
}


void do_PerspectiveProjection_2(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, cv::Mat &Mapping_RGB_Image, 
                              cv::Mat &Mapping_Depth_Image, float *viewpoint_Translation, float *viewpoint_Rotation,
                              std::vector<Point_with_Pixel> &PwPs, float intrinsic_fx, float intrinsic_fy, float intrinsic_cx, float intrinsic_cy)
{
  //Extrinsics parameters
  Eigen::Matrix4d cam1_H_world = Eigen::Matrix4d::Identity();
  
  intrinsic_parameters.at<float> (0, 0) = intrinsic_fx;
  intrinsic_parameters.at<float> (0, 1) = 0.0;
  intrinsic_parameters.at<float> (0, 2) = intrinsic_cx;
  intrinsic_parameters.at<float> (1, 0) = 0.0;
  intrinsic_parameters.at<float> (1, 1) = intrinsic_fy;
  intrinsic_parameters.at<float> (1, 2) = intrinsic_cy;
  intrinsic_parameters.at<float> (2, 0) = 0.0;
  intrinsic_parameters.at<float> (2, 1) = 0.0;
  intrinsic_parameters.at<float> (2, 2) = 1.0;

  //k1,k2,p1,p2,k3
  distortion_coefficient.at<float> (0) = 0.0;
  distortion_coefficient.at<float> (1) = 0.0;
  distortion_coefficient.at<float> (2) = 0.0;
  distortion_coefficient.at<float> (3) = 0.0;
  distortion_coefficient.at<float> (4) = 0.0;


  // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  Eigen::Matrix4d viewpoint_transform = Eigen::Matrix4d::Identity();

  do_ViewpointTrasform(viewpoint_transform, viewpoint_Translation, viewpoint_Rotation);
  
  cam1_H_world = viewpoint_transform;

  // Read 3D points: cloud-> vector
	std::vector<cv::Point3f> cloudPoints;
	CloudToVector(input_cloud, cloudPoints);
  
	cv::Vec3d cam1_H_world_rvec, cam1_H_world_tvec;
	Matrix4dToRodriguesTranslation(cam1_H_world, cam1_H_world_rvec, cam1_H_world_tvec);

	// Perspective Projection of Cloud Points to Image Plane
	std::vector<cv::Point2f> imagePoints;
	cv::projectPoints(cloudPoints, cam1_H_world_rvec, cam1_H_world_tvec, intrinsic_parameters, distortion_coefficient, imagePoints);

  Point_with_Pixel PwP;

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

      PwP.point = cloudPoints[i];
      PwP.pixel.x = idxX;
      PwP.pixel.y = idxY;
      PwPs.push_back(PwP);

      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[0] = (int)input_cloud->points[i].b;
      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[1] = (int)input_cloud->points[i].g;
      Mapping_RGB_Image.at<cv::Vec3b>(idxY, idxX)[2] = (int)input_cloud->points[i].r;

      z = input_cloud->points[i].z;
      if (z > 0.0 && z < 0.1)
      {
        z = z/0.1*255;
        Mapping_Depth_Image.at<uchar>(idxY, idxX) = round(z);
      }
    }
  }
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
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[0], Eigen::Vector3f::UnitX()));
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[1], Eigen::Vector3f::UnitY()));
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[2], Eigen::Vector3f::UnitZ()));

  // Apply an affine transform defined by an Eigen Transform.
  pcl::transformPointCloud (*input_cloud, *output_cloud, transform_trans);
  pcl::transformPointCloud (*output_cloud, *output_cloud, transform_rotate);

  transform_trans.translation() << pca_location[0], pca_location[1], pca_location[2];

  pcl::transformPointCloud (*output_cloud, *output_cloud, transform_trans);
}

void do_Rotate_2(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud,
               Eigen::Vector4f pca_location,
               float *Rotate_angle,
               float z_dist)
{
  Eigen::Affine3f transform_trans = Eigen::Affine3f::Identity();
  Eigen::Affine3f transform_rotate = Eigen::Affine3f::Identity();

  // Define a translation .
  transform_trans.translation() << -1*pca_location[0], -1*pca_location[1], -1*pca_location[2];

  // The same rotation matrix as before; theta radians around Z axis
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[0], Eigen::Vector3f::UnitX()));
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[1], Eigen::Vector3f::UnitY()));
  transform_rotate.rotate (Eigen::AngleAxisf (Rotate_angle[2], Eigen::Vector3f::UnitZ()));

  // Apply an affine transform defined by an Eigen Transform.
  pcl::transformPointCloud (*input_cloud, *output_cloud, transform_trans);
  pcl::transformPointCloud (*output_cloud, *output_cloud, transform_rotate);

  transform_trans.translation() << 0, 0, z_dist;

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
  sor.setLeafSize (0.001, 0.001, 0.001);   //設定體素網格的大小
  sor.filter (*output_cloud);      //儲存濾波後的點雲
}

void do_Callback_AngleAxisRotation(pcl_utils::AngleAxis_rotation_msg AngleAxis_rotation)
{
  Angle_axis_rotation_open = AngleAxis_rotation.x;
  Angle_axis_rotation_normal = AngleAxis_rotation.y;
}

void do_Callback_2DPredict_AngleAxisRotation(pcl_utils::AngleAxis_rotation_msg AngleAxis_rotation)
{
  Angle_axis_rotation_approach =  AngleAxis_rotation.z;
}

AxisQuaterniond do_AngelAxis(Eigen::Vector3d &open_vector, Eigen::Vector3d &approach_vector, Eigen::Vector3d &normal_vector,
                  float rotation_open, float rotation_approach, float rotation_normal)
{
  open_vector.normalize();
  approach_vector.normalize();
  normal_vector.normalize();

  // cout << "open_vector \n" << open_vector << "\n\n";
  // cout << "approach_vector \n" << approach_vector << "\n\n";
  // cout << "normal_vector \n" << normal_vector << "\n\n\n";

  Eigen::Matrix3d R_1, R_2, R_3, R_rotate;

  // X axis
  R_1 = Eigen::AngleAxisd(0 , Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(0 , Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(0 , Eigen::Vector3d::UnitX());
  
  // Y axis
  R_2 = Eigen::AngleAxisd(M_PI/2 , Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(0 , Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(0 , Eigen::Vector3d::UnitX());
  
  // Z axis
  R_3 = Eigen::AngleAxisd(0 , Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(-1*M_PI/2 , Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(0 , Eigen::Vector3d::UnitX());
  
  R_rotate = Eigen::AngleAxisd(rotation_approach, Eigen::Vector3d::UnitZ())
           * Eigen::AngleAxisd(rotation_normal , Eigen::Vector3d::UnitY())
           * Eigen::AngleAxisd(rotation_open, Eigen::Vector3d::UnitX()); 
            
  R_1 = R_rotate * R_1;
  R_2 = R_rotate * R_2;
  R_3 = R_rotate * R_3;

  open_vector = R_rotate * open_vector;
  approach_vector = R_rotate * approach_vector;
  normal_vector = R_rotate * normal_vector;

  // cout << "rotated open_vector \n" << open_vector << "\n\n";
  // cout << "rotated approach_vector \n" << approach_vector << "\n\n";
  // cout << "rotated normal_vector \n" << normal_vector << "\n\n\n";

  AxisQuaterniond AQ;

  Eigen::Quaterniond open_q(R_1);
  Eigen::Quaterniond normal_q(R_2);
  Eigen::Quaterniond approach_q(R_3);

  AQ.open_q = open_q;
  AQ.normal_q = normal_q;
  AQ.approach_q = approach_q;

  return AQ;
}

void do_calculate_normal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr
                                   ,pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals1)
{
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
  ne.setInputCloud (point_cloud_ptr);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (0.01);
  ne.compute (*cloud_normals1);
}

pcl_utils::coordinate_normal  average_normal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud
                                                  ,pcl::PointCloud<pcl::PointNormal>::Ptr input_normal){

  pcl_utils::coordinate_normal average_normal;
  average_normal.x=0;
  average_normal.y=0;
  average_normal.z=0;
  average_normal.normal_x = 0.0;
  average_normal.normal_y = 0.0;
  average_normal.normal_z = 0.0;
  int count = 0;

  for (size_t currentPoint = 0; currentPoint < input_normal->points.size(); currentPoint++)
	{
    if(!(isnan(input_normal->points[currentPoint].normal[0])||
         isnan(input_normal->points[currentPoint].normal[1])||
         isnan(input_normal->points[currentPoint].normal[2])))
    {
      average_normal.x = average_normal.x + input_cloud->points[currentPoint].x;
      average_normal.y = average_normal.y + input_cloud->points[currentPoint].y;
      average_normal.z = average_normal.z + input_cloud->points[currentPoint].z;
      average_normal.normal_x = average_normal.normal_x + input_normal->points[currentPoint].normal[0];
      average_normal.normal_y = average_normal.normal_y + input_normal->points[currentPoint].normal[1];
      average_normal.normal_z = average_normal.normal_z + input_normal->points[currentPoint].normal[2];
		  
      count++;
      // std::cout << "Point:" << std::endl;
      // cout << currentPoint << std::endl;
		  // std::cout << "\town:" << average_normal.x << " "
		  		                  // << average_normal.y << " "
		  		                  // << average_normal.z << std::endl;

		  // std::cout << "\tNormal:" << input_cloud->points[currentPoint].x << " "
		  // 		                      << input_cloud->points[currentPoint].y << " "
		  // 		                      << input_cloud->points[currentPoint].z << std::endl;
    }
  }
  // printf("================\ncloud_size = %d \n",count);
  if (count == 0)
  {
    // cout << "zero normal found!, input_normal->points.size()=" << input_normal->points.size() << endl;
    count = 1;
  }
  average_normal.x = average_normal.x/count;
  average_normal.y = average_normal.y/count;
  average_normal.z = average_normal.z/count;
  average_normal.normal_x = average_normal.normal_x/count;
  average_normal.normal_y = average_normal.normal_y/count;
  average_normal.normal_z = average_normal.normal_z/count;
  
  return average_normal;
}


bool do_calculate_number_of_pointcloud(cv::Point2f grcnn_predict, float angle, std::vector<Point_with_Pixel> &PwPs, 
                                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float *new_point, oan_vector &output_oan_vector)
{
  Eigen::Vector3d open_vector(1, 0, 0);
  Eigen::Vector3d approach_vector(0, 0, 1);
  Eigen::Vector3d normal_vector(0, 1, 0);

  float d_1 = 0, d_2 = 0, d_3 = 0;
  float h_1 = 0.085/2, h_2 = 0.037/2, h_3 = 0.021/2, finger_length = 0.02;
  float thr = 3;
  float right_finger_x = 0, right_finger_y = 0, right_finger_z = 0;

  open_vector(0) = cos(-1*angle);
  open_vector(1) = sin(-1*angle);

  normal_vector = approach_vector.cross(open_vector);

  AxisQuaterniond AQ;

  AQ = do_AngelAxis(open_vector, approach_vector, normal_vector, 
                    Angle_axis_rotation_open, 
                    Angle_axis_rotation_approach, 
                    Angle_axis_rotation_normal);
  
  
  for(int i = 0 ; i < PwPs.size() ; i ++)
  {
    if (abs(PwPs[i].pixel.x - grcnn_predict.x) < thr & abs(PwPs[i].pixel.y - grcnn_predict.y) < thr)//need to be check for more carefully! Maybe multi points can be projected to the same point!
    {
      // cout << "find!" << "\n";

      float point_x, point_y, point_z;
      int number_of_point = 0, number_of_point_finger = 0;

      point_x = PwPs[i].point.x;
      point_y = PwPs[i].point.y;
      point_z = PwPs[i].point.z;// + h_2 ;

      new_point [0] = point_x;
      new_point [1] = point_y;
      new_point [2] = point_z;

      d_1 = point_x * open_vector(0) + point_y * open_vector(1) + point_z * open_vector(2);
      d_2 = point_x * approach_vector(0) + point_y * approach_vector(1) + point_z * approach_vector(2);
      d_3 = point_x * normal_vector(0) + point_y * normal_vector(1) + point_z * normal_vector(2);

      grab_cloud->clear();
      grab_cloud_left->clear();
      grab_cloud_right->clear();

      float open_volume = 0;

      if(input_cloud->size()!= 0)
      {
        for (int i = 0; i < input_cloud->size(); i++)
        {
          if (abs(input_cloud->points[i].x*normal_vector(0) + input_cloud->points[i].y*normal_vector(1) + input_cloud->points[i].z*normal_vector(2) - d_3) < h_3)
          {
            if (abs(input_cloud->points[i].x*approach_vector(0) + input_cloud->points[i].y*approach_vector(1) + input_cloud->points[i].z*approach_vector(2) - d_2) < h_2)
            {

              open_volume = input_cloud->points[i].x*open_vector(0) + input_cloud->points[i].y*open_vector(1) + input_cloud->points[i].z*open_vector(2) - d_1;

              if (abs(open_volume) < h_1)
              {
                grab_cloud->push_back(input_cloud->points[i]);
                number_of_point++;
              }

              if (open_volume > 0)
              {
                if (open_volume < h_1)
                {
                  grab_cloud_left->push_back(input_cloud->points[i]);

                }
              }

              if (open_volume < 0)
              {
                if (open_volume > -1*h_1)
                {
                  grab_cloud_right->push_back(input_cloud->points[i]);
                }
              }
              
              if (abs(open_volume) < (h_1 + finger_length))
              {
                number_of_point_finger++;
              }

            }
          }
        }
      }

      std_msgs::Int64 grab_point_num, finger_grab_point_num;

      grab_point_num.data = number_of_point;
      finger_grab_point_num.data = number_of_point_finger - number_of_point;

      pubNumGrabPoint.publish(grab_point_num);

      pubNumFingerGrabPoint.publish(finger_grab_point_num);
      
      //=========rviz marker=========
      open_arrow.header.stamp = ros::Time();
      open_arrow.pose.position.x = point_x;
      open_arrow.pose.position.y = point_y;
      open_arrow.pose.position.z = point_z;
      open_arrow.pose.orientation.x = AQ.open_q.x();
      open_arrow.pose.orientation.y = AQ.open_q.y();
      open_arrow.pose.orientation.z = AQ.open_q.z();
      open_arrow.pose.orientation.w = AQ.open_q.w();
      open_arrow.scale.x = h_1;


      pubAngleAxisOpen.publish(open_arrow);

      normal_arrow.header.stamp = ros::Time();
      normal_arrow.pose.position.x = point_x;
      normal_arrow.pose.position.y = point_y;
      normal_arrow.pose.position.z = point_z;
      normal_arrow.pose.orientation.x = AQ.normal_q.x();
      normal_arrow.pose.orientation.y = AQ.normal_q.y();
      normal_arrow.pose.orientation.z = AQ.normal_q.z();
      normal_arrow.pose.orientation.w = AQ.normal_q.w();
      normal_arrow.scale.x = h_3;

      pubAngleAxisNormal.publish(normal_arrow);

      approach_arrow.header.stamp = ros::Time();
      approach_arrow.pose.position.x = point_x;
      approach_arrow.pose.position.y = point_y;
      approach_arrow.pose.position.z = point_z;
      approach_arrow.pose.orientation.x = AQ.approach_q.x();
      approach_arrow.pose.orientation.y = AQ.approach_q.y();
      approach_arrow.pose.orientation.z = AQ.approach_q.z();
      approach_arrow.pose.orientation.w = AQ.approach_q.w();
      approach_arrow.scale.x = h_2;

      pubAngleAxisApproach.publish(approach_arrow);

      //=========rviz marker=========

      output_oan_vector.open_vector(0) = open_vector(0);
      output_oan_vector.open_vector(1) = open_vector(1);
      output_oan_vector.open_vector(2) = open_vector(2);
      output_oan_vector.open_vector(3) = d_1;


      output_oan_vector.approach_vector(0) = approach_vector(0);
      output_oan_vector.approach_vector(1) = approach_vector(1);
      output_oan_vector.approach_vector(2) = approach_vector(2);
      output_oan_vector.approach_vector(3) = d_2;


      output_oan_vector.normal_vector(0) = normal_vector(0);
      output_oan_vector.normal_vector(1) = normal_vector(1);
      output_oan_vector.normal_vector(2) = normal_vector(2);
      output_oan_vector.normal_vector(3) = d_3;

      return true;
    }
  }
  return false;
}
void do_Callback_GrcnnResult(pcl_utils::grcnn_result result)
{
  grcnn_input.x = result.x;
  grcnn_input.y = result.y;
  grcnn_input.angle = result.angle;
  grcnn_input.length = result.length;
  grcnn_input.width = result.width;
}

void do_Project_using_parametric_model(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, 
                                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud,
                                       float *model_coefficients)
{
  // Create a set of planar coefficients
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  coefficients->values.resize (4);
  coefficients->values[0] = model_coefficients[0];
  coefficients->values[1] = model_coefficients[1];
  coefficients->values[2] = model_coefficients[2];
  coefficients->values[3] = model_coefficients[3];

  // Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZRGB> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  proj.setInputCloud (input_cloud);
  proj.setModelCoefficients (coefficients);
  proj.filter (*output_cloud);
}


void do_Callback_PointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{  
  // ROS to PCL
  pcl::fromROSMsg(*cloud_msg, *cloud);

  do_Passthrough(cloud, filter_cloud, "x", -0.28, 0.35);
  do_Passthrough(filter_cloud, filter_cloud, "y", -1, 0.08);
  do_Passthrough(filter_cloud, filter_cloud, "z", -1, z_passthrough);
  do_VoxelGrid(filter_cloud, filter_cloud);
  
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
  string Save_Data_path = "/home/luca/datasets/my_grasp_dataset/";
  string Name_pcd = "pcd";
  string Name_RGB_Image_root = "r.png";
  string Name_Depth_Image_root = "d.tiff";
  string Name_PCD_root = ".pcd";
  string Name_txt_root = ".txt";


  //save RGB image as .png format
  cv::imwrite(Save_Data_path + Name_pcd + SaveImage_Counter_Wrapper(take_picture_counter, req.call) + Name_RGB_Image_root, Mapping_RGB_Image);

  //save depth image as .tiff format
  cv::imwrite(Save_Data_path + Name_pcd + SaveImage_Counter_Wrapper(take_picture_counter, req.call) + Name_Depth_Image_root, Mapping_Depth_Image);

  pcl::io::savePCDFileASCII (Save_Data_path + Name_pcd + SaveImage_Counter_Wrapper(take_picture_counter, req.call) + Name_PCD_root, *filter_cloud);
  
  ofstream ofs;
  ofs.open(Save_Data_path + Name_pcd + SaveImage_Counter_Wrapper(take_picture_counter, req.call) + Name_txt_root);

  if (!ofs.is_open()) {
        cout << "Failed to open file.\n";
    } else {
        ofs << grcnn_input.x << "\n";
        ofs << grcnn_input.y << "\n";
        ofs << Angle_axis_rotation_approach;
        ofs.close();
  }
  cout << "Angle_axis_rotation_approach " << Angle_axis_rotation_approach <<endl;
  take_picture_counter++;

  res.back = take_picture_counter;

  ROS_INFO("Done SaveImage!");

  return true;
}

bool do_setPointCloud(pcl_utils::setPointCloud::Request &req, pcl_utils::setPointCloud::Response &res)
{
  cout<<"Set Input PointCloud"<<endl;
  
  pcl::copyPointCloud(*filter_cloud, *now_cloud);
  Set_Input_PointCloud = true;

  res.back = 0;
  return true;
}

bool do_setZPassthrough(pcl_utils::setZPassthrough::Request &req, pcl_utils::setZPassthrough::Response &res)
{
  z_passthrough = req.z_dist;

  ROS_INFO("Set Z-axis Passthrough distance!");
  res.back = 0;
  return true;
}

bool do_loadPointCloud(pcl_utils::loadPointCloud::Request &req, pcl_utils::loadPointCloud::Response &res)
{
  string Load_File_path = "/home/luca/datasets/GraspPointDataset/";
  string Name_pcd = "pcd99";
  string Name_PCD_root = ".pcd";
  string PCD_File_Name;
  string Name_txt_root = ".txt";

  string PCD_Num_string;

  if (nowLocalPCD <10)
  {
    PCD_Num_string = "0" + to_string(nowLocalPCD);
  }
  else
  {
    PCD_Num_string = to_string(nowLocalPCD);
  }

  PCD_File_Name = Load_File_path + Name_pcd + PCD_Num_string + Name_PCD_root;

  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (PCD_File_Name, *now_cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return false;
  }


  pcl::toROSMsg(*now_cloud, now_cloud_output);
  now_cloud_output.header.frame_id = "depth_camera_link";
  pubNowCloud.publish(now_cloud_output);

  ifstream ifs;
  string line;
  
  float predict_x, predict_y, predict_theta; 
  
  ifs.open(Load_File_path + Name_pcd + PCD_Num_string + Name_txt_root);

  getline(ifs, line);
  grcnn_input.x = stof(line);
  getline(ifs, line);
  grcnn_input.y = stof(line);
  getline(ifs, line);
  Angle_axis_rotation_approach = stof(line);

  // cout << "grcnn_input.x " << grcnn_input.x << ", grcnn_input.y " << grcnn_input.y << ", Angle_axis_rotation_approach " << Angle_axis_rotation_approach << "\n";

  // Close file
  ifs.close();

  nowLocalPCD++;

  if (nowLocalPCD == NumberOfLocalPCD)
  {
    nowLocalPCD = 0;
  }
  return true;
}

void do_PointcloudProcess()
{
  ros::WallTime start_, end_;
  start_ = ros::WallTime::now();

  //=== Get projected rgb & depth image form pointcloud === [begin]
  //reset values in the images
  Mapping_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
  Mapping_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));

  //get random number
  // float random_rotation = unif(generator);

  viewpoint_translation[0] = 0.0;
  viewpoint_translation[1] = 0.0;
  viewpoint_translation[2] = 0.0;
  viewpoint_rotation[0] = 0.0;
  viewpoint_rotation[1] = 0.0;
  viewpoint_rotation[2] = 0.0;


  if(filter_cloud->size()!=0)
  {
    std::vector<Point_with_Pixel> filter_cloud_PwPs;

    do_PerspectiveProjection(filter_cloud, Mapping_RGB_Image, Mapping_Depth_Image, viewpoint_translation, viewpoint_rotation, filter_cloud_PwPs, fx, fy, cx, cy);

    //do dilate for sparse image result
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));  
    cv::dilate(Mapping_RGB_Image, Mapping_RGB_Image, element);
    cv::dilate(Mapping_Depth_Image, Mapping_Depth_Image, element);

    sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", Mapping_RGB_Image).toImageMsg();
    ros::Time rgb_begin = ros::Time::now();
    rgb_msg->header.stamp = rgb_begin;
    pubProjectRGBImage.publish(rgb_msg);

    sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", Mapping_Depth_Image).toImageMsg();
    ros::Time depth_begin = ros::Time::now();
    depth_msg->header.stamp = depth_begin;
    pubProjectDepthImage.publish(depth_msg);
    //=== Get projected rgb & depth image form pointcloud === [end]
  }
  

  if(now_cloud->size()!=0)
  {
    //=== Get grab pointclout & count it's number === [begin]
    float grasp_angle = grcnn_input.angle;
    grasp_angle = 0;

    cv::Point2f grcnn_predict;

    grcnn_predict.x = grcnn_input.x;
    grcnn_predict.y = grcnn_input.y;

    oan_vector plane_coefficients_vector;

    // if(Set_Input_PointCloud)
    // {
    //   Set_Input_PointCloud = false;
    //   cout <<"Set_Input_PointCloud~~~"<<endl;
    // }
    std::vector<Point_with_Pixel> now_cloud_PwPs;

    do_PerspectiveProjection(now_cloud, Mapping_RGB_Image, Mapping_Depth_Image, viewpoint_translation, viewpoint_rotation, now_cloud_PwPs, fx, fy, cx, cy);

    if(do_calculate_number_of_pointcloud(grcnn_predict, grasp_angle, now_cloud_PwPs, now_cloud, grasp_3D, plane_coefficients_vector))
    {
      float *Rotate_angle = new float[3];

      Rotate_angle[0] = -1 * Angle_axis_rotation_open;  //X axis
      Rotate_angle[1] = -1 * Angle_axis_rotation_normal;  //Y axis
      Rotate_angle[2] = -1 * Angle_axis_rotation_approach;  //Z axis

      Eigen::Vector4f grasp_point;

      grasp_point(0) = grasp_3D[0];
      grasp_point(1) = grasp_3D[1];
      grasp_point(2) = grasp_3D[2];

      float z_dist = 0.05;
      //rotate point Cloud
      do_Rotate_2(grab_cloud, 
                retransform_approach_vector_plane_cloud, 
                grasp_point, 
                Rotate_angle,
                z_dist);

      Rotate_angle[0] = Rotate_angle[0]+M_PI/2;  //X axis

      //rotate point Cloud
      do_Rotate_2(grab_cloud, 
                retransform_normal_vector_plane_cloud, 
                grasp_point, 
                Rotate_angle,
                z_dist);

      Rotate_angle[0] = Rotate_angle[0]-M_PI/2;  //X axis
      Rotate_angle[1] = Rotate_angle[1]-M_PI/2;
      //rotate point Cloud
      do_Rotate_2(grab_cloud, 
                retransform_open_vector_plane_cloud, 
                grasp_point, 
                Rotate_angle,
                z_dist);

      pcl::toROSMsg(*retransform_normal_vector_plane_cloud, retransform_project_normal_vector_plane_output);
      retransform_project_normal_vector_plane_output.header.frame_id = "depth_camera_link";
      pubRetransformProjectNormalVectorPlaneCloud.publish(retransform_project_normal_vector_plane_output);

      pcl::toROSMsg(*retransform_approach_vector_plane_cloud, retransform_project_approach_vector_plane_output);
      retransform_project_approach_vector_plane_output.header.frame_id = "depth_camera_link";
      pubRetransformProjectApproachVectorPlaneCloud.publish(retransform_project_approach_vector_plane_output);

      pcl::toROSMsg(*retransform_open_vector_plane_cloud, retransform_project_open_vector_plane_output);
      retransform_project_open_vector_plane_output.header.frame_id = "depth_camera_link";
      pubRetransformProjectOpenVectorPlaneCloud.publish(retransform_project_open_vector_plane_output);

      pcl::toROSMsg(*grab_cloud, grab_output);
      grab_output.header.frame_id = "depth_camera_link";
      pubGrabPointClouds.publish(grab_output);

      pcl::toROSMsg(*grab_cloud_left, grab_output_left);
      grab_output_left.header.frame_id = "depth_camera_link";
      pubGrabPointCloudsLeft.publish(grab_output_left);

      pcl::toROSMsg(*grab_cloud_right, grab_output_right);
      grab_output_right.header.frame_id = "depth_camera_link";
      pubGrabPointCloudsRight.publish(grab_output_right);

      //Grab cloud normal
      if (grab_cloud->size()!= 0)
      {
        pcl::PointCloud<pcl::PointNormal>::Ptr grab_cloud_normal (new pcl::PointCloud<pcl::PointNormal>);
        do_calculate_normal(grab_cloud, grab_cloud_normal);
        pcl_utils::coordinate_normal object_normal_grab;
        object_normal_grab = average_normal(grab_cloud, grab_cloud_normal);
        pubGrabPointCloudsNormal.publish(object_normal_grab);

        geometry_msgs::PoseStamped object_pose;

        object_pose.header.frame_id = "depth_camera_link";
        object_pose.header.stamp = ros::Time::now();;
        object_pose.header.seq = 1;


        // extracting surface normals
        tf::Vector3 grab_axis_vector(object_normal_grab.normal_x, object_normal_grab.normal_y, object_normal_grab.normal_z);
        tf::Vector3 grab_up_vector(1.0, 0.0, 0.0);

        tf::Vector3 grab_right_vector = grab_axis_vector.cross(grab_up_vector);
        grab_right_vector.normalized();
        tf::Quaternion grab_q(grab_right_vector, -1.0*acos(grab_axis_vector.dot(grab_up_vector)));
        grab_q.normalize();

        geometry_msgs::Quaternion grab_msg;

        tf::quaternionTFToMsg(grab_q, grab_msg);

        object_pose.pose.orientation = grab_msg;
        object_pose.pose.position.x = object_normal_grab.x;
        object_pose.pose.position.y = object_normal_grab.y;
        object_pose.pose.position.z = object_normal_grab.z;

        pub_pose_grab.publish (object_pose);

        float normal_likelihood = -1.0 * (plane_coefficients_vector.approach_vector(0) * object_normal_grab.normal_x 
                                        + plane_coefficients_vector.approach_vector(1) * object_normal_grab.normal_y 
                                        + plane_coefficients_vector.approach_vector(2) * object_normal_grab.normal_z);
        std_msgs::Float64 normal_likelihood_msg;
        normal_likelihood_msg.data = normal_likelihood;
        pubNormalLikelihood.publish(normal_likelihood_msg);                                
      }
      
      //left finger====================================================================
      if(grab_cloud_left->size()!= 0)
      {
        pcl::PointCloud<pcl::PointNormal>::Ptr left_cloud_normal (new pcl::PointCloud<pcl::PointNormal>);
        do_calculate_normal(grab_cloud_left, left_cloud_normal);
        pcl_utils::coordinate_normal object_normal_left;
        object_normal_left = average_normal(grab_cloud_left, left_cloud_normal);
        pubGrabPointCloudsLeftNormal.publish(object_normal_left);

        geometry_msgs::PoseStamped object_pose_left;

        object_pose_left.header.frame_id = "depth_camera_link";
        object_pose_left.header.stamp = ros::Time::now();;
        object_pose_left.header.seq = 1;

        // extracting surface normals
        tf::Vector3 left_axis_vector(object_normal_left.normal_x, object_normal_left.normal_y, object_normal_left.normal_z);
        tf::Vector3 left_up_vector(1.0, 0.0, 0.0);

        tf::Vector3 left_right_vector = left_axis_vector.cross(left_up_vector);
        left_right_vector.normalized();
        tf::Quaternion left_q(left_right_vector, -1.0*acos(left_axis_vector.dot(left_up_vector)));
        left_q.normalize();

        geometry_msgs::Quaternion left_msg;

        tf::quaternionTFToMsg(left_q, left_msg);

        object_pose_left.pose.orientation = left_msg;
        object_pose_left.pose.position.x = object_normal_left.x;
        object_pose_left.pose.position.y = object_normal_left.y;
        object_pose_left.pose.position.z = object_normal_left.z;

        pub_pose_left.publish (object_pose_left);

        float left_likelihood = (plane_coefficients_vector.open_vector(0) * object_normal_left.normal_x 
                              + plane_coefficients_vector.open_vector(1) * object_normal_left.normal_y 
                              + plane_coefficients_vector.open_vector(2) * object_normal_left.normal_z);

        if (isnan(left_likelihood))
        {

          cout << " left_likelihood Not a Number FOUNDED!!!" <<endl;
        }

        std_msgs::Float64 left_likelihood_msg;
        left_likelihood_msg.data = left_likelihood;
        pubLeftLikelihood.publish(left_likelihood_msg);

      }

      //right finger====================================================================
      if (grab_cloud_right->size()!= 0)
      {
        pcl::PointCloud<pcl::PointNormal>::Ptr right_cloud_normal (new pcl::PointCloud<pcl::PointNormal>);
        do_calculate_normal(grab_cloud_right, right_cloud_normal);
        pcl_utils::coordinate_normal object_normal_right;
        object_normal_right = average_normal(grab_cloud_right, right_cloud_normal);

        pubGrabPointCloudsRightNormal.publish(object_normal_right);

        geometry_msgs::PoseStamped object_pose_right;

        object_pose_right.header.frame_id = "depth_camera_link";
        object_pose_right.header.stamp = ros::Time::now();;
        object_pose_right.header.seq = 1;
        
        geometry_msgs::Quaternion right_msg;

        // extracting surface normals
        tf::Vector3 right_axis_vector(object_normal_right.normal_x, object_normal_right.normal_y, object_normal_right.normal_z);
        tf::Vector3 right_up_vector(1.0, 0.0, 0.0);

        tf::Vector3 right_right_vector = right_axis_vector.cross(right_up_vector);
        right_right_vector.normalized();
        tf::Quaternion right_q(right_right_vector, -1.0*acos(right_axis_vector.dot(right_up_vector)));
        right_q.normalize();
        tf::quaternionTFToMsg(right_q, right_msg);

        object_pose_right.pose.orientation = right_msg;
        object_pose_right.pose.position.x = object_normal_right.x;
        object_pose_right.pose.position.y = object_normal_right.y;
        object_pose_right.pose.position.z = object_normal_right.z;
        
        pub_pose_right.publish (object_pose_right);

        float right_likelihood = (-1.0*plane_coefficients_vector.open_vector(0) * object_normal_right.normal_x 
                                + -1.0*plane_coefficients_vector.open_vector(1) * object_normal_right.normal_y 
                                + -1.0*plane_coefficients_vector.open_vector(2) * object_normal_right.normal_z);


        if (isnan(right_likelihood))
        {
          cout << " right_likelihood Not a Number FOUNDED!!!" <<endl;
        }
        
        std_msgs::Float64 right_likelihood_msg;
        right_likelihood_msg.data = right_likelihood;
        pubRightLikelihood.publish(right_likelihood_msg);
      }

      std_msgs::Float64 approach_likelihood_msg;
      // approach to (0, 0, 1) is better
      float approach_likelihood =  plane_coefficients_vector.approach_vector(2);
      if (isnan(approach_likelihood))
      {
        cout << " approach_likelihood Not a Number FOUNDED!!!" <<endl;
      }
      approach_likelihood_msg.data = approach_likelihood;
      //!!!!!!!!!!!!!!!!!
      pubApproachLikelihood.publish(approach_likelihood_msg);

      //====================================================================

      //=== publish plane pointcloud and grab pointcloud === [end]

      //=== Project plane Image === [begin]
      std::vector<Point_with_Pixel> Grab_Cloud_Approach_PwPs, Grab_Cloud_Normal_PwPs, Grab_Cloud_Open_PwPs;

      Grab_Cloud_viewpoint_Translation[0] = 0.0;
      Grab_Cloud_viewpoint_Translation[1] = 0.0;
      Grab_Cloud_viewpoint_Translation[2] = 0.0;

      Grab_Cloud_viewpoint_Rotation[0] = 0;
      Grab_Cloud_viewpoint_Rotation[1] = 0;
      Grab_Cloud_viewpoint_Rotation[2] = 0;
      
      Grab_Cloud_Approach_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
      Grab_Cloud_Approach_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));
      Grab_Cloud_Normal_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
      Grab_Cloud_Normal_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));
      Grab_Cloud_Open_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
      Grab_Cloud_Open_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));

      do_PerspectiveProjection_2(retransform_approach_vector_plane_cloud, Grab_Cloud_Approach_RGB_Image, Grab_Cloud_Approach_Depth_Image, 
                                Grab_Cloud_viewpoint_Translation, Grab_Cloud_viewpoint_Rotation, Grab_Cloud_Approach_PwPs,
                                Mapping_width/2, Mapping_high/2, 300, 300);
      
      do_PerspectiveProjection_2(retransform_open_vector_plane_cloud, Grab_Cloud_Open_RGB_Image, Grab_Cloud_Open_Depth_Image, 
                              Grab_Cloud_viewpoint_Translation, Grab_Cloud_viewpoint_Rotation, Grab_Cloud_Open_PwPs,
                              Mapping_width/2, Mapping_high/2, 300, 300);

      do_PerspectiveProjection_2(retransform_normal_vector_plane_cloud, Grab_Cloud_Normal_RGB_Image, Grab_Cloud_Normal_Depth_Image, 
                              Grab_Cloud_viewpoint_Translation, Grab_Cloud_viewpoint_Rotation, Grab_Cloud_Normal_PwPs,
                              Mapping_width/2, Mapping_high/2, 300, 300);

      cv::Mat Grab_element = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));  
      
      cv::dilate(Grab_Cloud_Approach_RGB_Image, Grab_Cloud_Approach_RGB_Image, Grab_element);
      cv::dilate(Grab_Cloud_Approach_Depth_Image, Grab_Cloud_Approach_Depth_Image, Grab_element); 

      cv::dilate(Grab_Cloud_Open_RGB_Image, Grab_Cloud_Open_RGB_Image, Grab_element);
      cv::dilate(Grab_Cloud_Open_Depth_Image, Grab_Cloud_Open_Depth_Image, Grab_element); 

      cv::dilate(Grab_Cloud_Normal_RGB_Image, Grab_Cloud_Normal_RGB_Image, Grab_element);
      cv::dilate(Grab_Cloud_Normal_Depth_Image, Grab_Cloud_Normal_Depth_Image, Grab_element); 

      // cv::Size cv_downsize = cv::Size(320, 240);
      cv::Size cv_downsize = cv::Size(160, 120);

      cv::resize(Grab_Cloud_Normal_RGB_Image, Grab_Cloud_Normal_RGB_Image, cv_downsize, cv::INTER_AREA);
      cv::resize(Grab_Cloud_Approach_RGB_Image, Grab_Cloud_Approach_RGB_Image, cv_downsize, cv::INTER_AREA);
      cv::resize(Grab_Cloud_Open_RGB_Image, Grab_Cloud_Open_RGB_Image, cv_downsize, cv::INTER_AREA);

      cv::resize(Grab_Cloud_Normal_Depth_Image, Grab_Cloud_Normal_Depth_Image, cv_downsize, cv::INTER_AREA);
      cv::resize(Grab_Cloud_Approach_Depth_Image, Grab_Cloud_Approach_Depth_Image, cv_downsize, cv::INTER_AREA);
      cv::resize(Grab_Cloud_Open_Depth_Image, Grab_Cloud_Open_Depth_Image, cv_downsize, cv::INTER_AREA);

      int NormaldepthNonZeroValue = cv::countNonZero(Grab_Cloud_Normal_Depth_Image);
      int OpendepthNonZeroValue = cv::countNonZero(Grab_Cloud_Open_Depth_Image);

      // cout << "Number of non-zero image Grab_Cloud_Normal_Depth_Image: " << NormaldepthNonZeroValue << endl;

      std_msgs::Float64 NormaldepthNonZeroValue_msg;
      NormaldepthNonZeroValue_msg.data = NormaldepthNonZeroValue;
      pubNormaldepthNonZero.publish(NormaldepthNonZeroValue_msg);

      std_msgs::Float64 OpendepthNonZeroValue_msg;
      OpendepthNonZeroValue_msg.data = OpendepthNonZeroValue;
      pubOpendepthNonZero.publish(OpendepthNonZeroValue_msg);

      //=== publish mapping image === [begin]
      sensor_msgs::ImagePtr grab_approach_rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", Grab_Cloud_Approach_RGB_Image).toImageMsg();
      ros::Time grab_approach_rgb_begin = ros::Time::now();
      grab_approach_rgb_msg->header.stamp = grab_approach_rgb_begin;
      pubProject_Grab_Approach_RGB_Image.publish(grab_approach_rgb_msg);

      sensor_msgs::ImagePtr grab_approach_depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", Grab_Cloud_Approach_Depth_Image).toImageMsg();
      ros::Time grab_approach_depth_begin = ros::Time::now();
      grab_approach_depth_msg->header.stamp = grab_approach_depth_begin;
      pubProject_Grab_Approach_Depth_Image.publish(grab_approach_depth_msg);

      sensor_msgs::ImagePtr grab_open_rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", Grab_Cloud_Open_RGB_Image).toImageMsg();
      ros::Time grab_open_rgb_begin = ros::Time::now();
      grab_open_rgb_msg->header.stamp = grab_open_rgb_begin;
      pubProject_Grab_Open_RGB_Image.publish(grab_open_rgb_msg);

      sensor_msgs::ImagePtr grab_open_depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", Grab_Cloud_Open_Depth_Image).toImageMsg();
      ros::Time grab_open_depth_begin = ros::Time::now();
      grab_open_depth_msg->header.stamp = grab_approach_depth_begin;
      pubProject_Grab_Open_Depth_Image.publish(grab_open_depth_msg);

      sensor_msgs::ImagePtr grab_normal_rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", Grab_Cloud_Normal_RGB_Image).toImageMsg();
      ros::Time grab_normal_rgb_begin = ros::Time::now();
      grab_normal_rgb_msg->header.stamp = grab_normal_rgb_begin;
      pubProject_Grab_Normal_RGB_Image.publish(grab_normal_rgb_msg);

      sensor_msgs::ImagePtr grab_normal_depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", Grab_Cloud_Normal_Depth_Image).toImageMsg();
      ros::Time grab_normal_depth_begin = ros::Time::now();
      grab_normal_depth_msg->header.stamp = grab_normal_depth_begin;
      pubProject_Grab_Normal_Depth_Image.publish(grab_normal_depth_msg);
      //=== publish image === [end]
    }
  }
  end_ = ros::WallTime::now();
  // print results
  double execution_time = (end_ - start_).toNSec() * 1e-6;
  if (execution_time > max_execution_time)
  {
    max_execution_time = execution_time;
  }
  // ROS_INFO_STREAM("Exectution time (ms): " << execution_time);
  // max_execution_time = max_execution_time -0.1;
  // ROS_INFO_STREAM("max_execution_time time (ms): " << max_execution_time);

}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pcl_service");

  // Initialize NodeHandle
  ros::NodeHandle nh;

  // Creat marker for rviz
  open_arrow.header.frame_id = "depth_camera_link";
  open_arrow.ns = "my_namespace";
  open_arrow.id = 0;
  open_arrow.type = visualization_msgs::Marker::ARROW;
  open_arrow.action = visualization_msgs::Marker::ADD;
  open_arrow.scale.y = 0.003;
  open_arrow.scale.z = 0.003;
  open_arrow.color.a = 1.0; // Don't forget to set the alpha!
  open_arrow.color.r = 1.0;
  open_arrow.color.g = 0.0;
  open_arrow.color.b = 0.0;

  normal_arrow.header.frame_id = "depth_camera_link";
  normal_arrow.ns = "my_namespace";
  normal_arrow.id = 2;
  normal_arrow.type = visualization_msgs::Marker::ARROW;
  normal_arrow.action = visualization_msgs::Marker::ADD;
  normal_arrow.scale.y = 0.003;
  normal_arrow.scale.z = 0.003;
  normal_arrow.color.a = 1.0; // Don't forget to set the alpha!
  normal_arrow.color.r = 0.0;
  normal_arrow.color.g = 1.0;
  normal_arrow.color.b = 0.0;

  approach_arrow.header.frame_id = "depth_camera_link";
  approach_arrow.ns = "my_namespace";
  approach_arrow.id = 1;
  approach_arrow.type = visualization_msgs::Marker::ARROW;
  approach_arrow.action = visualization_msgs::Marker::ADD;
  approach_arrow.scale.y = 0.003;
  approach_arrow.scale.z = 0.003;
  approach_arrow.color.a = 1.0; // Don't forget to set the alpha!
  approach_arrow.color.r = 0.0;
  approach_arrow.color.g = 0.0;
  approach_arrow.color.b = 1.0;

  right_finger_point.header.frame_id = "depth_camera_link";
  right_finger_point.ns = "my_namespace";
  right_finger_point.id = 1;
  right_finger_point.type = visualization_msgs::Marker::SPHERE;
  right_finger_point.action = visualization_msgs::Marker::ADD;
  right_finger_point.scale.x = 0.01;
  right_finger_point.scale.y = 0.01;
  right_finger_point.scale.z = 0.01;
  right_finger_point.color.a = 1.0; // Don't forget to set the alpha!
  right_finger_point.color.r = 1.0;
  right_finger_point.color.g = 1.0;
  right_finger_point.color.b = 0.0;

  // Create ROS publisher for marker in rviz
  pubAngleAxisOpen = nh.advertise<visualization_msgs::Marker>("/pubAngleAxisOpen", 0);
  pubAngleAxisApproach = nh.advertise<visualization_msgs::Marker>("/pubAngleAxisApproach", 0);
  pubAngleAxisNormal = nh.advertise<visualization_msgs::Marker>("/pubAngleAxisNormal", 0);
  pubRightFingerPoint = nh.advertise<visualization_msgs::Marker>("/pubRightFingerPoint", 0);
  
  // Create ROS publisher for projected image
  image_transport::ImageTransport it(nh);
  pubProjectRGBImage = it.advertise("/projected_image/rgb", 1);
  pubProjectDepthImage = it.advertise("/projected_image/depth", 1);
  pubProject_Grab_Approach_RGB_Image = it.advertise("/projected_image/grab_approach_rgb", 1);
  pubProject_Grab_Approach_Depth_Image = it.advertise("/projected_image/grab_approach_depth", 1);
  pubProject_Grab_Normal_RGB_Image = it.advertise("/projected_image/grab_normal_rgb", 1);
  pubProject_Grab_Normal_Depth_Image = it.advertise("/projected_image/grab_normal_depth", 1);
  pubProject_Grab_Open_RGB_Image = it.advertise("/projected_image/grab_open_rgb", 1);
  pubProject_Grab_Open_Depth_Image = it.advertise("/projected_image/grab_open_depth", 1);

  // Create ROS pointcloud publisher for the rotate point cloud
  pubRotatePointClouds = nh.advertise<sensor_msgs::PointCloud2> ("/Rotate_PointClouds", 30);

  // Create ROS pointcloud publisher for the point cloud of gripped area
  pubGrabPointClouds = nh.advertise<sensor_msgs::PointCloud2> ("/Grab_PointClouds", 30);

  pubGrabPointCloudsLeft = nh.advertise<sensor_msgs::PointCloud2> ("/Grab_PointClouds_Left", 30);

  pubGrabPointCloudsRight = nh.advertise<sensor_msgs::PointCloud2> ("/Grab_PointClouds_Right", 30);

  pubGrabPointCloudsNormal = nh.advertise<pcl_utils::coordinate_normal> ("/Grab_PointClouds_Normal", 1);

  pubGrabPointCloudsLeftNormal = nh.advertise<pcl_utils::coordinate_normal> ("/Grab_PointClouds_Left_Normal", 1);

  pubGrabPointCloudsRightNormal = nh.advertise<pcl_utils::coordinate_normal> ("/Grab_PointClouds_Right_Normal", 1);

  pub_pose_grab = nh.advertise<geometry_msgs::PoseStamped> ("/object/pose/grab", 1);

  pub_pose_left = nh.advertise<geometry_msgs::PoseStamped> ("/object/pose/left", 1);
  
  pub_pose_right = nh.advertise<geometry_msgs::PoseStamped> ("/object/pose/right", 1);
  
  // Create ROS pointcloud publisher for the number of grab point
  pubNumGrabPoint = nh.advertise<std_msgs::Int64> ("/Number_of_Grab_PointClouds", 30);

  // Create ROS pointcloud publisher for the number of finger grab point
  pubNumFingerGrabPoint = nh.advertise<std_msgs::Int64> ("/Number_of_Finger_Grab_PointClouds", 30);

  pubNormalLikelihood = nh.advertise<std_msgs::Float64> ("/PointLikelihood/Grab_Cloud", 30);

  pubLeftLikelihood = nh.advertise<std_msgs::Float64> ("/PointLikelihood/Left_Finger", 30);

  pubRightLikelihood = nh.advertise<std_msgs::Float64> ("/PointLikelihood/Right_Finger", 30);

  pubApproachLikelihood = nh.advertise<std_msgs::Float64> ("/ApporachLikelihood", 30);

  pubNormaldepthNonZero = nh.advertise<std_msgs::Float64> ("/NormaldepthNonZero", 30);

  pubOpendepthNonZero = nh.advertise<std_msgs::Float64> ("/OpendepthNonZero", 30);

  // Create ROS pointcloud publisher for projected normal vector plane cloud
  pubProjectNormalVectorPlaneCloud = nh.advertise<sensor_msgs::PointCloud2> ("/Project_Normal_Vector_PlaneClouds", 30);

  // Create ROS pointcloud publisher for projected approach vector plane cloud
  pubProjectApproachVectorPlaneCloud = nh.advertise<sensor_msgs::PointCloud2> ("/Project_Approach_Vector_PlaneClouds", 30);

  // Create ROS pointcloud publisher for projected open vector plane cloud
  pubProjectOpenVectorPlaneCloud = nh.advertise<sensor_msgs::PointCloud2> ("/Project_Open_Vector_PlaneClouds", 30);

  // Create ROS pointcloud publisher for retransform projected normal vector plane cloud
  pubRetransformProjectNormalVectorPlaneCloud= nh.advertise<sensor_msgs::PointCloud2> ("/Retransform_project_Normal_Vector_PlaneClouds", 30);

  // Create ROS pointcloud publisher for retransform projected approach vector plane cloud
  pubRetransformProjectApproachVectorPlaneCloud= nh.advertise<sensor_msgs::PointCloud2> ("/Retransform_project_Approach_Vector_PlaneClouds", 30);

  // Create ROS pointcloud publisher for retransform projected open vector plane cloud
  pubRetransformProjectOpenVectorPlaneCloud= nh.advertise<sensor_msgs::PointCloud2> ("/Retransform_project_Open_Vector_PlaneClouds", 30);

  pubNowCloud = nh.advertise<sensor_msgs::PointCloud2> ("/Now_Clouds", 30);


  // Create ROS subscriber for the input point cloud (azure kinect dk)
  ros::Subscriber subSaveCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/points2", 1, do_Callback_PointCloud);

  // Create ROS subscriber for the result of grcnn (2D grasp point)
  ros::Subscriber subGrcnnResult = nh.subscribe<pcl_utils::grcnn_result> ("/grcnn/result", 1, do_Callback_GrcnnResult);

  // Create ROS subscriber for the AngleAxis_rotation (open, normal & approach)
  ros::Subscriber subAngleAxisRotation = nh.subscribe<pcl_utils::AngleAxis_rotation_msg> ("/grasp_training/AngleAxis_rotation", 1, do_Callback_AngleAxisRotation);

  ros::Subscriber sub2DPredictAngleAxisRotation = nh.subscribe<pcl_utils::AngleAxis_rotation_msg> ("/2D_Predict/AngleAxis_rotation", 1, do_Callback_2DPredict_AngleAxisRotation);


  // Create ROS Service for taking picture
  ros::ServiceServer saveImage_service = nh.advertiseService("/snapshot", do_SaveImage);

  ros::ServiceServer set_input_PointCloud_service = nh.advertiseService("/set_pointcloud", do_setPointCloud);

  ros::ServiceServer load_input_PointCloud_service = nh.advertiseService("/load_pointcloud", do_loadPointCloud);

  ros::ServiceServer set_z_passthrough_service = nh.advertiseService("/set_z_passthrough", do_setZPassthrough);


  ros::WallTime start_, end_;

  ros::Rate loop_rate(100);

  while(ros::ok())
  {
    do_PointcloudProcess();
    ros::spinOnce();
    // loop_rate.sleep();
  }

  return 0;
}
