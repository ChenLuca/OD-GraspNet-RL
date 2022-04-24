#!/usr/bin/env python3
import sys
import rospy
import cv2
import numpy as np
import math

from skimage.feature import peak_local_max

sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

# from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import detect_grasps
from grcnn.msg import dl_grasp_result
from grcnn.msg import AngleAxis_rotation_msg
from pcl_utils.srv import snapshot
logging.basicConfig(level=logging.INFO)

rgb_bridge = CvBridge()
depth_bridge = CvBridge()

global_q_img = np.zeros((0,0,1), np.uint8)
global_ang_img = np.zeros((0,0,1), np.uint8)

rgb_image = np.zeros((0,0,3), np.uint8)
depth_image = np.zeros((0,0,1), np.uint8)

save_img_counter = 0
no_grasps = 1
loc_old_trained_custom_data = '/home/ur5/code/RL-Grasp-with-GRCNN/src/grcnn/scripts/trained-models/my_model/20210921/epoch_33_iou_0.65'
loc_grcnn = '/home/ur5/code/RL-Grasp-with-GRCNN/src/grcnn/scripts/trained-models/my_model/default/epoch_19_iou_0.98'
loc_ODR_ConvNet_v4 = '/home/ur5/code/RL-Grasp-with-GRCNN/src/grcnn/scripts/trained-models/211123_0713_ODR_ConvNet_3_VoV/epoch_16_iou_0.92'
loc_OD_ConvNet_v1_dilated = '/home/ur5/code/RL-Grasp-with-GRCNN/src/grcnn/scripts/trained-models/211123_1729_OD_ConvNet_1_dilated/epoch_19_iou_0.92'
loc_OD_ConvNet_3_csp = "/home/ur5/code/RL-Grasp-with-GRCNN/src/grcnn/scripts/trained-models/211127_1303_OD_ConvNet_3_csp/epoch_10_iou_0.97"

loc_grcnn_J = "/home/ur5/code/RL-Grasp-with-GRCNN/src/grcnn/scripts/trained-models/grcnn_jac/epoch_39_iou_0.90"
loc_ODR_ConvNet_v1_IM_J = "/home/ur5/code/RL-Grasp-with-GRCNN/src/grcnn/scripts/trained-models/odr_conv1_im_jac/epoch_82_iou_0.91"

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default=loc_grcnn_J,
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

def rgb_callback(image):
    global rgb_image
    try:
        rgb_image = rgb_bridge.imgmsg_to_cv2(image, "rgb8")
    except CvBridgeError as e:
        print(e)

def depth_callback(image):
    global depth_image
    try:
        depth_image = depth_bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        print(e)


        
def q_img_sample(q_img, ang_img):

    def normalize_data(ori_data):

        sum_data = sum(ori_data)
        norm_data = ori_data/sum_data

        return norm_data

    # Step1: sample n local_max point
    local_max = peak_local_max(q_img, min_distance=8, threshold_abs=0.2, num_peaks=10)

    # Step2: get gray_scale value and normalized to 1
    val_ori = []
    for i in range(len(local_max)):
        val = q_img[local_max[i][0],local_max[i][1]]
        print(local_max[i][0],local_max[i][1], val)
        val_ori.append(val)

    val_normalized = normalize_data(val_ori)
    print('val_normalized', val_normalized)

    # Step3: sample again from n local_max point,
    #        the quality(0-255) of image is the possibility to be sampled
    point_idx = [i for i in range(len(local_max))]
    print("point_idx", point_idx)
    print("val_normalized ", val_normalized)
    print("local_max ", local_max)
    sampled_point_idx = np.random.choice(point_idx, 1, p=val_normalized)

    # output: sampled pixel position
    print("pixel coord: ", local_max[sampled_point_idx][0], "grasp quality: ", val_ori[sampled_point_idx[0]])

def handle_get_q_ang_image(req):
    
    print("in service get_q_image, req:", req)

    global save_img_counter
    cv2.imwrite("/home/ur5/datasets/GraspPointDataset/saving/q_img_{}.png".format(save_img_counter), q_img*255)
    # cv2.imwrite("/home/ur5/datasets/GraspPointDataset/saving/ang_img_{}.png".format(save_img_counter), ang_img)
    np.save("/home/ur5/datasets/GraspPointDataset/saving/ang_img_{}".format(save_img_counter), ang_img)
    save_img_counter = save_img_counter + 1

    q_img_sample(global_q_img, ang_img)

    return 99

if __name__ == '__main__':

    args = parse_args()

    rospy.init_node('grcnn_inference', anonymous=True)

    pub_AngleAxisRotation = rospy.Publisher('/2D_Predict/AngleAxis_rotation', AngleAxis_rotation_msg, queue_size=10)

    pub_osa_result = rospy.Publisher('/dl_grasp/result', dl_grasp_result, queue_size=10)

    rospy.Subscriber("/projected_image/rgb", Image, rgb_callback)

    rospy.Subscriber("/projected_image/depth", Image, depth_callback)

    s = rospy.Service('get_q_image', snapshot, handle_get_q_ang_image)

    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    while not rospy.is_shutdown():
        try:
            fig = plt.figure(figsize=(10, 10))
            while True:
                rgb = rgb_image
                depth = np.expand_dims(depth_image, axis=2)
                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

                with torch.no_grad():
                    xc = x.to(device)
                    pred = net.predict(xc)

                    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                    
                    global_q_img = q_img
                    global_ang_img = ang_img

                    gs = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=no_grasps)

                    if gs is not None:
                        for g in gs:
                            osa_result_msg = dl_grasp_result()
                            osa_result_msg.y = g.center[0] + 110
                            osa_result_msg.x = g.center[1] + 190
                            osa_result_msg.angle = g.angle
                            osa_result_msg.length = g.length
                            osa_result_msg.width = g.width
                            pub_osa_result.publish(osa_result_msg)

                            print("center(y, x):{}, angle:{}, length:{}, width:{} ".format(g.center, g.angle, g.length, g.width))

                            rotation = AngleAxis_rotation_msg()
                            rotation.x = 0
                            rotation.y = 0
                            rotation.z = -1* g.angle 
                            pub_AngleAxisRotation.publish(rotation)

                    plot_results(fig=fig,
                                rgb_img=cam_data.get_rgb(rgb, False),
                                depth_img=np.squeeze(cam_data.get_depth(depth)),
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                no_grasps=args.n_grasps,
                                grasp_width_img=width_img)
        finally:
            print('bye grcnn_inference!')
