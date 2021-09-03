import os
import glob
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='', help='path to Trainingdata')
FLAGS = parser.parse_args()
dataset_path = FLAGS.dataset_path

# generic location of this file
Generic_location = os.path.abspath(__file__ + "/..")

# select data type & sort
grasp_rgb = glob.glob(os.path.join(Generic_location + "/" + dataset_path, '*', 'pcd*r.png'))
grasp_rgb.sort()

# vurr image data
curr_rgb_image = np.zeros((0, 0, 3), dtype=np.float32)
curr_image_name = "none"

# graphical
drawing = False
Line_Point_1 = [0, 0]
Line_Point_2 = [0, 0]

# dats storage
Line_Point_1_data =[]
Line_Point_2_data =[]
P1_data = []
P2_data = []
P3_data = []
P4_data = []

def draw(event, x, y, flags, param):

    global curr_image_name, curr_rgb_image, drawing, Line_Point_1, Line_Point_2
    
    painted_image = curr_rgb_image.copy()

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        
        # set line point 1
        Line_Point_1[0], Line_Point_1[1] = x,y
        Line_Point_1_data.append([Line_Point_1[0], Line_Point_1[1]])

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            # draw line 
            cv2.line(painted_image, (Line_Point_1[0], Line_Point_1[1]), (x, y), (0,0,255), 3)
            
            # draw center of line
            running_center_x = int((Line_Point_1[0] + x)/2)
            running_center_y = int((Line_Point_1[1] + y)/2)
            cv2.circle(painted_image, (running_center_x, running_center_y), 5, (0, 255, 255), 3)
            
            # cover old image, for draw just one line on an image
            cv2.imshow(curr_image_name, painted_image)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False
        
        # set line point 2
        Line_Point_2[0], Line_Point_2[1] = x,y
        Line_Point_2_data.append([Line_Point_2[0], Line_Point_2[1]])

        # compute four points of grasp rectangle
        P1, P2, P3, P4 = compute_four_points(Line_Point_1, Line_Point_2)
        
        # append data
        P1_data.append(P1)
        P2_data.append(P2)
        P3_data.append(P3)
        P4_data.append(P4)

def compute_four_points(Line_Point_1, Line_Point_2):
    
    # calculate difference of x & y
    dx = Line_Point_2[0] - Line_Point_1[0]
    dy = Line_Point_2[1] - Line_Point_1[1]
    
    theta = (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    gripper_width = 20

    w_cos = (gripper_width * np.cos(theta + np.pi/2))
    w_sin = (gripper_width * np.sin(theta + np.pi/2))

    P1 = [ int(Line_Point_1[0] + w_cos), int(Line_Point_1[1] - w_sin) ]
    P2 = [ int(Line_Point_2[0] + w_cos), int(Line_Point_2[1] - w_sin) ]
    P3 = [ int(Line_Point_2[0] - w_cos), int(Line_Point_2[1] + w_sin) ]
    P4 = [ int(Line_Point_1[0] - w_cos), int(Line_Point_1[1] + w_sin) ]

    print("===========")
    print("[dx dy] [{}, {}] ".format(dx, dy) )
    print("theta ", theta)
    print("[w_cos w_sin] [{}, {}] ".format(w_cos, w_sin) )
    print("Line_Point_1 ", Line_Point_1)
    print("Line_Point_2 ", Line_Point_2)
    print("P1 ", P1)
    print("P2 ", P2)
    print("P3 ", P3)
    print("P4 ", P4)
    print("===========")

    return P1, P2, P3, P4

def SaveRectangle():
    
    with open( curr_image_name.replace("r.png", "cpos.txt"), 'a') as f:
        
        for i in range(len(P1_data)):
            
            if i == (len(P1_data)-1):
                line_1 = str(P1_data[i][0]) + " " + str(P1_data[i][1]) + "\n"
                line_2 = str(P2_data[i][0]) + " " + str(P2_data[i][1]) + "\n"
                line_3 = str(P3_data[i][0]) + " " + str(P3_data[i][1]) + "\n"
                line_4 = str(P4_data[i][0]) + " " + str(P4_data[i][1])
            else:
                line_1 = str(P1_data[i][0]) + " " + str(P1_data[i][1]) + "\n"
                line_2 = str(P2_data[i][0]) + " " + str(P2_data[i][1]) + "\n"
                line_3 = str(P3_data[i][0]) + " " + str(P3_data[i][1]) + "\n"
                line_4 = str(P4_data[i][0]) + " " + str(P4_data[i][1]) + "\n"

            f.writelines(line_1)
            f.writelines(line_2)
            f.writelines(line_3)
            f.writelines(line_4)

def loop():
    
    global curr_image_namee

    cv2.namedWindow(curr_image_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(curr_image_name, draw)

    painted_image = curr_rgb_image.copy()
    
    while 1:
        
        global Line_Point_1_data, Line_Point_2_data, curr_rgb_imag, P1, P2, P3, P4

        k = cv2.waitKey(1) & 0xFF

        # cancelled crop
        if (k == 27):
            print("Cancelled Crop")
            break

        # remove last rectangle
        elif (k == ord('c')):

            # pop list data
            if len(Line_Point_2_data) != 0:
                
                Line_Point_1_data.pop()
                Line_Point_2_data.pop()
                
                P1_data.pop()
                P2_data.pop()
                P3_data.pop()
                P4_data.pop()
                
                # cover to origin image
                painted_image = curr_rgb_image.copy()
            pass

        # save file
        elif(k == ord('s')):
            
            SaveRectangle()
            print("================Saved=================")
            print("Data : ", curr_image_name)
            print("Data point 1 : ", P1_data)
            print("Data point 2 : ", P2_data)
            print("Data point 3 : ", P3_data)
            print("Data point 4 : ", P4_data)

            print("Number of Label :", len(P1_data))
            print("======================================\n")
            Line_Point_1_data.clear()
            Line_Point_2_data.clear()
            P1_data.clear()
            P2_data.clear()
            P3_data.clear()
            P4_data.clear()
            
            break

        # draw picture
        if (len(Line_Point_2_data) != 0):
            for i in range(len(Line_Point_2_data)):

                cv2.line(painted_image, (Line_Point_1_data[i][0], Line_Point_1_data[i][1]), (Line_Point_2_data[i][0], Line_Point_2_data[i][1]), (0,0,255), 3)
                
                static_center_x = abs(int((Line_Point_2_data[i][0] + Line_Point_1_data[i][0])/2))
                static_center_y = abs(int((Line_Point_2_data[i][1] + Line_Point_1_data[i][1])/2))
                cv2.circle(painted_image, (static_center_x, static_center_y), 5, (0, 255, 255), 3)

                cv2.line(painted_image, (Line_Point_1_data[i][0], Line_Point_1_data[i][1]), (P1_data[i][0], P1_data[i][1]), (255, 0, 0), 5)
                cv2.line(painted_image, (Line_Point_2_data[i][0], Line_Point_2_data[i][1]), (P2_data[i][0], P2_data[i][1]), (0, 255, 0), 5)
                cv2.line(painted_image, (Line_Point_2_data[i][0], Line_Point_2_data[i][1]), (P3_data[i][0], P3_data[i][1]), (255, 255, 255), 5)
                cv2.line(painted_image, (Line_Point_1_data[i][0], Line_Point_1_data[i][1]), (P4_data[i][0], P4_data[i][1]), (255, 255, 255), 5)

                cv2.putText(painted_image, "P1", (P1_data[i][0], P1_data[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(painted_image, "P2", (P2_data[i][0], P2_data[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(painted_image, "P3", (P3_data[i][0], P3_data[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(painted_image, "P4", (P4_data[i][0], P4_data[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(curr_image_name, painted_image)

    cv2.destroyAllWindows()

def main():

    global curr_image_name, curr_rgb_image

    # if got data folder
    if len(grasp_rgb) > 0 :
        
        for i in range(len(grasp_rgb)):
        
            curr_image_name = grasp_rgb[i]
            curr_rgb_image = cv2.imread(grasp_rgb[i])

            loop()

if __name__ == "__main__":

    main()

    cv2.destroyAllWindows()
