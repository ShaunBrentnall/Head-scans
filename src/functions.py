#!/usr/bin/env python
# coding: utf-8

import numpy as np
from itertools import groupby

def scan_check(angles, angle_threshold, frame_threshold):
    
    scan_left = [1 if angles[i] >= angle_threshold else 0 for i in range(len(angles))]
    scan_right = [1 if -angles[i] >= angle_threshold else 0 for i in range(len(angles))]
    
    scan_left = [np.sum(list(group)) for key, group in groupby(scan_left)]
    scan_right = [np.sum(list(group)) for key, group in groupby(scan_right)]
    
    scan_left = list(filter(lambda x: x >= frame_threshold, scan_left))
    scan_right = list(filter(lambda x: x >= frame_threshold, scan_right))
    
    print("The total number of head scans above {}° was {}.\nLeft: {}\nRight: {}\nThe average time of a head scan was {:.2f} seconds.".format(angle_threshold, len(scan_left) + len(scan_right) , len(scan_left), len(scan_right), np.mean(scan_left + scan_right)/24))
    
    return #scan_left, scan_right