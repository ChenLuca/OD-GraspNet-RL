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

  std::vector<Point_with_Pixel> PwPs;

  do_PerspectiveProjection(filter_cloud, Mapping_RGB_Image, Mapping_Depth_Image, viewpoint_translation, viewpoint_rotation, PwPs, fx, fy, cx, cy);

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

  //=== Get grab pointclout & count it's number === [begin]
  float grasp_angle = grcnn_input.angle;
  grasp_angle = 0;

  cv::Point2f grcnn_predict;

  // grcnn_predict.x = grcnn_input.x;
  // grcnn_predict.y = grcnn_input.y;
  grcnn_predict.x = 320;
  grcnn_predict.y = 260;

  oan_vector plane_coefficients_vector;

  if(do_calculate_number_of_pointcloud(grcnn_predict, grasp_angle, PwPs, filter_cloud, grasp_3D, plane_coefficients_vector))
  {
    //=== Project plane pointcloud of grab_cloud from normal, approach and open vector === [end]
    // ax + by + cy + d = 0
    // plane_coefficients = a, b, c, d
    float *normal_vector_plane_coefficients = new float[4];
    float *approach_vector_plane_coefficients = new float[4];
    float *open_vector_plane_coefficients = new float[4];

    normal_vector_plane_coefficients[0] = plane_coefficients_vector.normal_vector(0);
    normal_vector_plane_coefficients[1] = plane_coefficients_vector.normal_vector(1);
    normal_vector_plane_coefficients[2] = plane_coefficients_vector.normal_vector(2);
    normal_vector_plane_coefficients[3] = -1*plane_coefficients_vector.normal_vector(3);

    approach_vector_plane_coefficients[0] = plane_coefficients_vector.approach_vector(0);
    approach_vector_plane_coefficients[1] = plane_coefficients_vector.approach_vector(1);
    approach_vector_plane_coefficients[2] = plane_coefficients_vector.approach_vector(2);
    approach_vector_plane_coefficients[3] = -1*plane_coefficients_vector.approach_vector(3);

    open_vector_plane_coefficients[0] = plane_coefficients_vector.open_vector(0);
    open_vector_plane_coefficients[1] = plane_coefficients_vector.open_vector(1);
    open_vector_plane_coefficients[2] = plane_coefficients_vector.open_vector(2);
    open_vector_plane_coefficients[3] = -1*plane_coefficients_vector.open_vector(3);

    do_Project_using_parametric_model(grab_cloud, project_normal_vector_plane_cloud, normal_vector_plane_coefficients);
    do_Project_using_parametric_model(grab_cloud, project_approach_vector_plane_cloud, approach_vector_plane_coefficients);
    do_Project_using_parametric_model(grab_cloud, project_open_vector_plane_cloud, open_vector_plane_coefficients);
    //=== Project plane pointcloud of grab_cloud from normal, approach and open vector === [end]

    float *Rotate_angle = new float[3];

    Rotate_angle[0] = -1 * Angle_axis_rotation_open;  //X axis
    Rotate_angle[1] = -1 * Angle_axis_rotation_normal;  //Y axis
    Rotate_angle[2] = -1 * Angle_axis_rotation_approach;  //Z axis

    Eigen::Vector4f grasp_point;

    grasp_point(0) = grasp_3D[0];
    grasp_point(1) = grasp_3D[1];
    grasp_point(2) = grasp_3D[2];

    //rotate point Cloud
    do_Rotate(project_normal_vector_plane_cloud, 
              retransform_normal_vector_plane_cloud, 
              grasp_point, 
              Rotate_angle);

    //rotate point Cloud
    do_Rotate(project_approach_vector_plane_cloud, 
              retransform_approach_vector_plane_cloud, 
              grasp_point, 
              Rotate_angle);

    //rotate point Cloud
    do_Rotate(project_open_vector_plane_cloud, 
              retransform_open_vector_plane_cloud, 
              grasp_point, 
              Rotate_angle);

    //=== publish plane pointcloud and grab pointcloud === [begin]
    pcl::toROSMsg(*project_normal_vector_plane_cloud, project_normal_vector_plane_output);
    project_normal_vector_plane_output.header.frame_id = "depth_camera_link";
    pubProjectNormalVectorPlaneCloud.publish(project_normal_vector_plane_output);

    pcl::toROSMsg(*project_approach_vector_plane_cloud, project_approach_vector_plane_output);
    project_approach_vector_plane_output.header.frame_id = "depth_camera_link";
    pubProjectApproachVectorPlaneCloud.publish(project_approach_vector_plane_output);

    pcl::toROSMsg(*project_open_vector_plane_cloud, project_open_vector_plane_output);
    project_open_vector_plane_output.header.frame_id = "depth_camera_link";
    pubProjectOpenVectorPlaneCloud.publish(project_open_vector_plane_output);

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


    pcl::PointCloud<pcl::PointNormal>::Ptr left_cloud_normal (new pcl::PointCloud<pcl::PointNormal>);
    do_calculate_normal(grab_cloud_left, left_cloud_normal);
    pcl_utils::coordinate_normal object_normal_left;
    object_normal_left = average_normal(grab_cloud_left, left_cloud_normal);

    pubGrabPointCloudsLeftNormal.publish(object_normal_left);

    geometry_msgs::PoseStamped object_pose_left;

    // object_pose_left.header.frame_id = "base_link";
    object_pose_left.header.frame_id = "depth_camera_link";
    object_pose_left.header.stamp = ros::Time::now();;
    object_pose_left.header.seq = 1;
    
    geometry_msgs::Quaternion left_msg;

    // extracting surface normals
    tf::Vector3 left_axis_vector(object_normal_left.normal_x, object_normal_left.normal_y, object_normal_left.normal_z);
    tf::Vector3 left_up_vector(1.0, 0.0, 0.0);

    tf::Vector3 left_right_vector = left_axis_vector.cross(left_up_vector);
    left_right_vector.normalized();
    tf::Quaternion left_q(left_right_vector, -1.0*acos(left_axis_vector.dot(left_up_vector)));
    left_q.normalize();
    tf::quaternionTFToMsg(left_q, left_msg);

    object_pose_left.pose.orientation = left_msg;
    object_pose_left.pose.position.x = object_normal_left.x;
    object_pose_left.pose.position.y = object_normal_left.y;
    object_pose_left.pose.position.z = object_normal_left.z;

    pub_pose_left.publish (object_pose_left);


    cout << "object_normal_left: " << object_normal_left.normal_x << ", " << object_normal_left.normal_y << ", " << object_normal_left.normal_z <<"\n\n";
    cout << "open vector: " << plane_coefficients_vector.open_vector(0) << ", " << plane_coefficients_vector.open_vector(1) << ", " << plane_coefficients_vector.open_vector(2) <<"\n\n";

    float left_likelihood = (plane_coefficients_vector.open_vector(0) * object_normal_left.normal_x + plane_coefficients_vector.open_vector(1) * object_normal_left.normal_y + plane_coefficients_vector.open_vector(2) * object_normal_left.normal_z);
    cout << "left_likelihood: " << left_likelihood << "\n\n";


    //right finger====================================================================
    pcl::PointCloud<pcl::PointNormal>::Ptr right_cloud_normal (new pcl::PointCloud<pcl::PointNormal>);
    do_calculate_normal(grab_cloud_right, right_cloud_normal);
    pcl_utils::coordinate_normal object_normal_right;
    object_normal_right = average_normal(grab_cloud_right, right_cloud_normal);

    pubGrabPointCloudsRightNormal.publish(object_normal_right);

    geometry_msgs::PoseStamped object_pose_right;

    // object_pose_right.header.frame_id = "base_link";
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


    cout << "object_normal_right: " << object_normal_right.normal_x << ", " << object_normal_right.normal_y << ", " << object_normal_right.normal_z <<"\n\n";

    float right_likelihood = (-1.0*plane_coefficients_vector.open_vector(0) * object_normal_right.normal_x 
                            + -1.0*plane_coefficients_vector.open_vector(1) * object_normal_right.normal_y 
                            + -1.0*plane_coefficients_vector.open_vector(2) * object_normal_right.normal_z);

    cout << "right_likelihood: " << right_likelihood << "\n\n";

    cout << "plane_coefficients_vector.approach_vector " << plane_coefficients_vector.approach_vector(0) << ", " << plane_coefficients_vector.approach_vector(1) << ", " << plane_coefficients_vector.approach_vector(2) << endl;

    // approach to (0, 0, 1) is better
    float approach_likelihood =  plane_coefficients_vector.approach_vector(2);

    cout << "approach_likelihood " << approach_likelihood << "\n";

    std_msgs::Float64 left_likelihood_msg, right_likelihood_msg, approach_likelihood_msg;

    left_likelihood_msg.data = left_likelihood;
    right_likelihood_msg.data = right_likelihood;
    approach_likelihood_msg.data = approach_likelihood;
    pubLeftLikelihood.publish(left_likelihood_msg);
    pubRightLikelihood.publish(right_likelihood_msg);

    //!!!!!!!!!!!!!!!!!
    pubApproachLikelihood.publish(approach_likelihood_msg);
    //====================================================================

    //=== publish plane pointcloud and grab pointcloud === [end]

    //=== Project plane Image === [begin]
    std::vector<Point_with_Pixel> Grab_Cloud_Approach_PwPs, Grab_Cloud_Normal_PwPs, Grab_Cloud_Open_PwPs;

    Grab_Cloud_viewpoint_Translation[0] = -grasp_3D[0];
    Grab_Cloud_viewpoint_Translation[1] = -grasp_3D[1];
    Grab_Cloud_viewpoint_Translation[2] = -grasp_3D[2] + 0.025;

    Grab_Cloud_viewpoint_Rotation[0] = 0;
    Grab_Cloud_viewpoint_Rotation[1] = 0;
    Grab_Cloud_viewpoint_Rotation[2] = 0;
    
    Grab_Cloud_Approach_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
    Grab_Cloud_Approach_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));
    Grab_Cloud_Normal_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
    Grab_Cloud_Normal_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));
    Grab_Cloud_Open_RGB_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC3, cv::Scalar(0, 0, 0));
    Grab_Cloud_Open_Depth_Image = cv::Mat(Mapping_high, Mapping_width, CV_8UC1, cv::Scalar(0));

    float *Rotate_Normal_Angle = new float[3];
    float *Rotate_Open_Angle = new float[3];

    Rotate_Normal_Angle[0] = M_PI/2;  //X axis
    Rotate_Normal_Angle[1] = 0;  //Y axis
    Rotate_Normal_Angle[2] = 0;  //Z axis

    Rotate_Open_Angle[0] = M_PI/2;  //X axis
    Rotate_Open_Angle[1] = 0;  //Y axis
    Rotate_Open_Angle[2] = M_PI/2;  //Z axis

      //rotate point Cloud
    do_Rotate(retransform_normal_vector_plane_cloud, 
              retransform_normal_vector_plane_cloud, 
              grasp_point, 
              Rotate_Normal_Angle);

    //rotate point Cloud
    do_Rotate(retransform_open_vector_plane_cloud, 
              retransform_open_vector_plane_cloud, 
              grasp_point, 
              Rotate_Open_Angle);

    do_PerspectiveProjection(retransform_approach_vector_plane_cloud, Grab_Cloud_Approach_RGB_Image, Grab_Cloud_Approach_Depth_Image, 
                              Grab_Cloud_viewpoint_Translation, Grab_Cloud_viewpoint_Rotation, Grab_Cloud_Approach_PwPs,
                              Mapping_width/2, Mapping_high/2, 300, 300);
    
    do_PerspectiveProjection(retransform_open_vector_plane_cloud, Grab_Cloud_Open_RGB_Image, Grab_Cloud_Open_Depth_Image, 
                            Grab_Cloud_viewpoint_Translation, Grab_Cloud_viewpoint_Rotation, Grab_Cloud_Open_PwPs,
                            Mapping_width/2, Mapping_high/2, 300, 300);

    do_PerspectiveProjection(retransform_normal_vector_plane_cloud, Grab_Cloud_Normal_RGB_Image, Grab_Cloud_Normal_Depth_Image, 
                            Grab_Cloud_viewpoint_Translation, Grab_Cloud_viewpoint_Rotation, Grab_Cloud_Normal_PwPs,
                            Mapping_width/2, Mapping_high/2, 300, 300);

    cv::Mat Grab_element = getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40));  
    
    cv::dilate(Grab_Cloud_Approach_RGB_Image, Grab_Cloud_Approach_RGB_Image, Grab_element);
    cv::dilate(Grab_Cloud_Approach_Depth_Image, Grab_Cloud_Approach_Depth_Image, Grab_element); // <===應該是平的！！！

    cv::dilate(Grab_Cloud_Open_RGB_Image, Grab_Cloud_Open_RGB_Image, Grab_element);
    cv::dilate(Grab_Cloud_Open_Depth_Image, Grab_Cloud_Open_Depth_Image, Grab_element); // <===應該是平的！！！

    cv::dilate(Grab_Cloud_Normal_RGB_Image, Grab_Cloud_Normal_RGB_Image, Grab_element);
    cv::dilate(Grab_Cloud_Normal_Depth_Image, Grab_Cloud_Normal_Depth_Image, Grab_element); // <===應該是平的！！！
    //=== Project Image === [end]

    // cv::Size cv_downsize = cv::Size(320, 240);
    cv::Size cv_downsize = cv::Size(160, 120);

    cv::resize(Grab_Cloud_Normal_RGB_Image, Grab_Cloud_Normal_RGB_Image, cv_downsize, cv::INTER_AREA);
    cv::resize(Grab_Cloud_Approach_RGB_Image, Grab_Cloud_Approach_RGB_Image, cv_downsize, cv::INTER_AREA);
    cv::resize(Grab_Cloud_Open_RGB_Image, Grab_Cloud_Open_RGB_Image, cv_downsize, cv::INTER_AREA);

    cv::resize(Grab_Cloud_Normal_Depth_Image, Grab_Cloud_Normal_Depth_Image, cv_downsize, cv::INTER_AREA);
    cv::resize(Grab_Cloud_Approach_Depth_Image, Grab_Cloud_Approach_Depth_Image, cv_downsize, cv::INTER_AREA);
    cv::resize(Grab_Cloud_Open_Depth_Image, Grab_Cloud_Open_Depth_Image, cv_downsize, cv::INTER_AREA);

    
    if(SHOW_CV_WINDOWS)
    {
      cv::namedWindow("Depth Image window", cv::WINDOW_AUTOSIZE);
      cv::imshow("Depth Image window", Mapping_Depth_Image);
      cv::namedWindow("RGB Image window", cv::WINDOW_AUTOSIZE);
      cv::imshow("RGB Image window", Mapping_RGB_Image);

      cv::namedWindow("Grab_Cloud_Depth_Image window", cv::WINDOW_AUTOSIZE);
      cv::imshow("Grab_Cloud_Depth_Image window", Grab_Cloud_Approach_Depth_Image);
      cv::namedWindow("Grab_Cloud_RGB_Image window", cv::WINDOW_AUTOSIZE);
      cv::imshow("Grab_Cloud_RGB_Image window", Grab_Cloud_Approach_RGB_Image);
      cv::waitKey(1);
    }


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

  //===  Get grab pointclout & count it's number === [end]

  ROS_INFO("Done do_Callback_PointCloud");
