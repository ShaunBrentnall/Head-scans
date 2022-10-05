#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
import math
import os

from functions import scan_check
import plotting

#Setup
#If raw_data returns nothing, print("Here's the link: ")

try:
    raw_data = pd.read_csv("data\\pose_joints.csv")
except Exception:
    print("File not found, try retrieving it again from here: https://drive.google.com/file/d/1j9poTVoDSv9GTSDC0SlROg6ZSdQ8u_GP/view?usp=sharing")

pd.set_option('display.max_columns', None)
df = pd.DataFrame(data = raw_data)


#Video statistics
WIDTH, HEIGHT = 1920, 1080
FRAMES = len(df)

#Shoulders
#----------------------------------------------------------------------------------
#Original vector place, V. O_x,y lies in the bottom left corner of the screen. 
#O_z is the midpoint between the hips (+z is in the forward direction of the player's body.) 
#PROBLEM: MAY HAVE TO SPLIT X, Y COORDINATES FROM Z COORDINATES. Z IS RADIAL DISTANCE FROM MH. 

LE = np.array(df.left_eye.apply(literal_eval))
RE = np.array(df.right_eye.apply(literal_eval))
ME = np.array([np.add(LE[i], RE[i])/2 for i in range(FRAMES)])

LS = np.array(df.left_shoulder.apply(literal_eval))
RS = np.array(df.right_shoulder.apply(literal_eval))
MS = np.array([np.add(LS[i], RS[i])/2 for i in range(FRAMES)])

#Transformation to new vector space, V_S. O_S_x,y lies at the midpoint between the shoulder. 
#T: V -> V_S
O_S = MS

LS_S = np.array([np.subtract(LS[i], MS[i]) for i in range(FRAMES)])
RS_S = np.array([np.subtract(RS[i], MS[i]) for i in range(FRAMES)])

LE_S = np.array([np.subtract(LE[i], MS[i]) for i in range(FRAMES)])
RE_S = np.array([np.subtract(RE[i], MS[i]) for i in range(FRAMES)])

D_S = np.array([np.subtract(ME[i], MS[i]) for i in range(FRAMES)]) #Directional vector between midpoint of eyes and midpoint of shoulders

#Orthogonal vectors
WE_S = np.array([np.cross(LE_S[i], D_S[i]) for i in range(FRAMES)])
WS_S = np.array([np.cross(LS_S[i], D_S[i]) for i in range(FRAMES)])

right_eye_shoulder = [np.sign(np.dot(RE_S[i], WS_S[i])) for i in range(FRAMES)] #The sign of the dot product determines if the payer is looking left or right relative to the body. Negative indicates looking to the right. 
angles_eye_shoulders= [right_eye_shoulder[i] * 90 * np.arccos(np.dot(WE_S[i], WS_S[i]) / (np.linalg.norm(WE_S[i]) * np.linalg.norm(WS_S[i]))) for i in range(FRAMES)]

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

DH_H = np.array([np.subtract(ME[i], MH[i]) for i in range(FRAMES)]) #Directional vector between midpoint of hips and midpoint of shoulders

#Orthogonal vectors
WE_H = np.array([np.cross(LE_H[i], DH_H[i]) for i in range(FRAMES)])
WH_H = np.array([np.cross(LH_H[i], DH_H[i]) for i in range(FRAMES)])

right_eye_hip = [np.sign(np.dot(RE_H[i], WH_H[i])) for i in range(FRAMES)]
angles_eye_hip = [right_eye_hip[i] * 90 * np.arccos(np.dot(WE_H[i], WH_H[i]) / (np.linalg.norm(WE_H[i]) * np.linalg.norm(WH_H[i]))) for i in range(FRAMES)]

#Results
#----------------------------------------------------------
#Head scan statistics
scan_check(angles_eye_hip, 30, 4)
scan_check(angles_eye_shoulders, 20, 4)

plotting.time_series_angle(angles_eye_shoulders, angles_eye_hip)

#Video
#----------------------------------------------
video_recording_path = os.path.join('videos','head_scan_1.mp4')
plotting.vid_angles(video_recording_path, angles_eye_hip)
