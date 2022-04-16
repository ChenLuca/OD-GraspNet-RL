#include <ros/ros.h>
#include "pcl_utils/snapshot.h"

using namespace std;

int main (int argc, char** argv)
{
  ros::init(argc, argv, "get_cornell_data");
  ros::NodeHandle nh;
  ros::ServiceClient saveImage_client = nh.serviceClient<pcl_utils::snapshot>("snapshot");
  ros::ServiceClient saveQImage_client = nh.serviceClient<pcl_utils::snapshot>("get_q_image");


  pcl_utils::snapshot snapshot_srv;
  int data_series_number = atoll(argv[1]);

  cout << "data series nember! : " << data_series_number <<endl;

  if (data_series_number > 10 and data_series_number < 100)
  {
    snapshot_srv.request.call = data_series_number;
    saveImage_client.call(snapshot_srv);
    saveQImage_client.call(snapshot_srv);
    
    cout << "save " << snapshot_srv.response.back << " pictures " <<endl;
  }
  else
  {
    cout << "no data saved, data series nember must between 11~99 !!" << endl;
  }

  return 0;
}