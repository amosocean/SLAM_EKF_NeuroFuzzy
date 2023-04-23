#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :EKF_SLAM.py
# @Time      :2023/4/14 8:38 PM
# @Author    :Kinddle
import os.path

import numpy as np

from utils import *
from rootFunctions import prediction_step,correction_step
# load sources
landmarks = read_world("sources/world.dat")     # 地标信息
data = read_data("sources/sensor_data.dat")
plot_path = "output/"

INF = 9999

N = len(landmarks)
observedLandmarks = np.repeat([False],N)

# init belief
mu = np.repeat(0,2*N+3)
robSigma = np.zeros([3,3])
robMapSigma = np.zeros([3,2*N])
mapSigma = np.eye(2*N) * INF
sigma = np.r_[np.c_[robSigma,robMapSigma],np.c_[robMapSigma.T,mapSigma]]

Flag_show_GUI = True
frames = []
for t in range(len(data)):
    mu, sigma = prediction_step(mu,sigma, data[t]["Odometry"],t==0)
    mu, sigma, observedLandmarks = correction_step(mu,sigma,data[t]["Sensor"],observedLandmarks)
    of = plot_state(mu, sigma, landmarks, t, observedLandmarks, data[t]["Sensor"], Flag_show_GUI)
    frames.append(of)
    # print("Current state vector:")
    print(f"\r simulating... {t+1}/{len(data)}",end="")


print("Final system covariance matrix:"), print(sigma)
# Display the final state estimate
print("Final robot pose:")
print("mu_robot = ")
print(mu[:3])
print("sigma_robot = ")
print(sigma[:3,:3])
print("saving...",end="")
frames[0].save(os.path.join(plot_path,"EKF_SLAM.gif"),save_all=True,loop=True, append_images=frames[1:],
               duration=10, disposal=2)
print("\rsaved!")
