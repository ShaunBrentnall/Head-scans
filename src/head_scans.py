#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import groupby
import cv2
import math

def scan_check(angles, angle_threshold, frame_threshold):
    
    scan_left = [1 if angles[i] >= angle_threshold else 0 for i in range(len(angles))]
    scan_right = [1 if -angles[i] >= angle_threshold else 0 for i in range(len(angles))]
    
    scan_left = [np.sum(list(group)) for key, group in groupby(scan_left)]
    scan_right = [np.sum(list(group)) for key, group in groupby(scan_right)]
    
    scan_left = list(filter(lambda x: x >= frame_threshold, scan_left))
    scan_right = list(filter(lambda x: x >= frame_threshold, scan_right))
    
    print("The total number of head scans above {}° was {}.\nLeft: {}\nRight: {}\nThe average time of a head scan was {:.2f} seconds.".format(angle_threshold, len(scan_left) + len(scan_right) , len(scan_left), len(scan_right), np.mean(scan_left + scan_right)/24))
    
    return #scan_left, scan_right



raw_data = pd.read_csv("C:\\Coding\\Projects\\head_scan\\pose_joints.csv")
pd.set_option('display.max_columns', None)
df = pd.DataFrame(data = raw_data)


#Video statistics
WIDTH, HEIGHT = 1920, 1080
FRAMES = len(df)

#Apply thresholds
THRESH_1 = 20
THRESH_2 = 40
THRESH_3 = 60

FRAME_THRESH = 4


#Shoulders
#----------------------------------------------------------------------------------

#Original vector place, V. O_x,y lies in the bottom left corner of the screen. 
#O_z is the midpoint between the hips (+z is either forward or backward direction of the player's body.) 
#PROBLEM: MAY HAVE TO SPLIT X, Y COORDINATES FROM Z COORDINATES. Z IS RADIAL DISTANCE FROM MH. 

LE = np.array(df.left_eye.apply(literal_eval))
RE = np.array(df.right_eye.apply(literal_eval))
ME = np.array([np.add(LE[i], RE[i])/2 for i in range(FRAMES)])


LS = np.array(df.left_shoulder.apply(literal_eval))
RS = np.array(df.right_shoulder.apply(literal_eval))
MS = np.array([np.add(LS[i], RS[i])/2 for i in range(FRAMES)])

N = np.array(df.nose.apply(literal_eval))

D = np.array([np.subtract(ME[i], MS[i]) for i in range(FRAMES)])


#Transformation to new vector space, V_S. O_S_x,y lies at the midpoint between the shoulder. 
#T: V -> V_S

O_S = MS

LS_S = np.array([np.subtract(LS[i], MS[i]) for i in range(FRAMES)])
RS_S = np.array([np.subtract(RS[i], MS[i]) for i in range(FRAMES)])

LE_S = np.array([np.subtract(LE[i], MS[i]) for i in range(FRAMES)])
RE_S = np.array([np.subtract(RE[i], MS[i]) for i in range(FRAMES)])

D_S = np.array([np.subtract(ME[i], MS[i]) for i in range(FRAMES)])


#Orthogonal vectors
WE_S = np.array([np.cross(LE_S[i], D[i]) for i in range(FRAMES)])
WS_S = np.array([np.cross(LS_S[i], D[i]) for i in range(FRAMES)])


right_eye = [np.sign(np.dot(RE_S[i], WS_S[i])) for i in range(FRAMES)]
angles_eye = [right_eye[i] * 90 * np.arccos(np.dot(WE_S[i], WS_S[i]) / (np.linalg.norm(WE_S[i]) * np.linalg.norm(WS_S[i]))) for i in range(FRAMES)]


#Hips
#----------------------------------------------------------------------------------

#Original vector place, V. O_x,y lies in the bottom left corner of the screen. 
#O_z is the midpoint between the hips (+z is either forward or backward direction of the player's body.) 
#PROBLEM: MAY HAVE TO SPLIT X, Y COORDINATES FROM Z COORDINATES. Z IS RADIAL DISTANCE FROM MH. O_H = MH

LH = np.array(df.left_hip.apply(literal_eval))
RH = np.array(df.right_hip.apply(literal_eval))
MH = np.array([np.add(LH[i], RH[i])/2 for i in range(FRAMES)])


#Transformation to new vector space, V_H. O_H_x,y lies at the midpoint between the hips. 
#T_H: V -> V_H

O_H = MH

LH_H = np.array([np.subtract(LH[i], MH[i]) for i in range(FRAMES)])
RH_H = np.array([np.subtract(RH[i], MH[i]) for i in range(FRAMES)])

LE_H = np.array([np.subtract(LE[i], MH[i]) for i in range(FRAMES)])
RE_H = np.array([np.subtract(RE[i], MH[i]) for i in range(FRAMES)])

DH_H = np.array([np.subtract(ME[i], MH[i]) for i in range(FRAMES)])


#Orthogonal vectors
WE_H = np.array([np.cross(LE_H[i], DH_H[i]) for i in range(FRAMES)])
WH_H = np.array([np.cross(LH_H[i], DH_H[i]) for i in range(FRAMES)])


right_eye_hip = [np.sign(np.dot(RE_H[i], WH_H[i])) for i in range(FRAMES)]
angles_eye_hip = [right_eye_hip[i] * 90 * np.arccos(np.dot(WE_H[i], WH_H[i]) / (np.linalg.norm(WE_H[i]) * np.linalg.norm(WH_H[i]))) for i in range(FRAMES)]


#Results
#----------------------------------------------------------
#Head scan statistics
scan_check(angles_eye_hip, 30, 4)
scan_check(angles_eye, 20, 4)


#Plots
figure(figsize=(20, 10), dpi=100)
"""
fig, ax = plt.subplots(2, 1, figsize = (8,10))

ax[0, 0].plot(angles_eye)

ax[1, 0].plot(angles_eye_hip)

fig.tight_layout()
"""
#fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(angles_eye)
plt.title("Angle between shoulder and eye direction")
plt.ylim(-90, 90)
plt.xlabel("Frame")
plt.ylabel("Head angle (°) \N{LEFTWARDS ARROW} right | left \N{RIGHTWARDS ARROW}")
plt.axhline(y=20, color='r', linestyle ='--')
plt.axhline(y=-20, color='r', linestyle ='--')


plt.subplot(2, 1, 2)
plt.plot(angles_eye_hip)
plt.title("Angle between hip and eye direction")
plt.ylim(-150, 150)
plt.xlabel("Frame")
plt.ylabel("Head angle (°) \N{LEFTWARDS ARROW} right | left \N{RIGHTWARDS ARROW}")
plt.axhline(y=30, color='r', linestyle ='--')
plt.axhline(y=-30, color='r', linestyle ='--')

plt.show()

#Video
#----------------------------------------------
"""
vid_capture = cv2.VideoCapture("C:\\Coding\\Projects\\head_scan\\head_scan_lampard.mp4")

if vid_capture.isOpened() == False:
    print("Error opening video")

else:
    fps = vid_capture.get(5)
    print("Frames per second: {}".format(fps))

    frame_count = vid_capture.get(7)
    print("Frame count: {}".format(frame_count))

    width = cv2.CAP_PROP_FRAME_WIDTH
    print("Frame width: {}".format(width))

wait = math.floor(1/fps * 75)


fig = plt.figure()


while vid_capture.isOpened():
    ret, frame = vid_capture.read()

    frame_number = int(vid_capture.get(cv2.CAP_PROP_POS_FRAMES))
  
    if ret == True:
        #figure(figsize=(20, 2), dpi=100)
        plt.scatter(angles_eye_hip[frame_number + 2],0)
        plt.ylim(-1,1)
        plt.xlim(150,-150)
        plt.xlabel("Head Angle (°): \N{LEFTWARDS ARROW} left | right \N{RIGHTWARDS ARROW}")
        plt.title("Angle between head and hip")
        fig.canvas.draw()
        
        graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))


        cv2.imshow("Frame", frame)
        cv2.imshow("plot", graph)

        #v_img = cv2.vconcat([frame, graph])
        #cv2.imshow("Head scans", v_img)

        key = cv2.waitKey(wait)

        if key == 113:  #113 is ASCII code for q key
            break
        plt.cla()
    else:
        break
#----------------------------------------------
"""
   



