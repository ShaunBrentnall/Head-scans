#!/usr/bin/env python
# coding: utf-8

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

#Plots

def time_series_angle(shoulders, hips):
    figure(figsize=(20, 10), dpi=100)

    #fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(shoulders)
    plt.title("Angle between shoulder and eye direction")
    plt.ylim(-90, 90)
    plt.xlabel("Frame")
    plt.ylabel("Head angle (°) \N{LEFTWARDS ARROW} right | left \N{RIGHTWARDS ARROW}")
    plt.axhline(y=20, color='r', linestyle ='--')
    plt.axhline(y=-20, color='r', linestyle ='--')


    plt.subplot(2, 1, 2)
    plt.plot(hips)
    plt.title("Angle between hip and eye direction")
    plt.ylim(-150, 150)
    plt.xlabel("Frame")
    plt.ylabel("Head angle (°) \N{LEFTWARDS ARROW} right | left \N{RIGHTWARDS ARROW}")
    plt.axhline(y=30, color='r', linestyle ='--')
    plt.axhline(y=-30, color='r', linestyle ='--')

    plt.show()


def vid_angles(video, angles):
    vid_capture = cv2.VideoCapture(video)

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
            plt.scatter(angles[frame_number + 2],0)
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