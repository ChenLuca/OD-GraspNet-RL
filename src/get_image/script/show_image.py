#!/usr/bin/env python3
import sys
import rospy
import cv2
import numpy as np

sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

print("python version ", sys.version)

class Get_image():
    def __init__(self):
        rospy.init_node('get_image_from_Azure_Kinect_DK', anonymous=True)
        
        self.bridge = CvBridge()
        self.image = np.zeros((0,0,3), np.uint8)

        rospy.Subscriber("/projected_image/rgb", Image, self.callback)

        rospy.spin()

    def callback(self, image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", self.cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    listener = Get_image()
    cv2.destroyAllWindows()
