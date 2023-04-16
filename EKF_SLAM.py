#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :EKF_SLAM.py
# @Time      :2023/4/14 8:38 PM
# @Author    :Kinddle
import numpy as np

from tools import *
from rootFunctions import prediction_step,correction_step
# load data
landmark = read_world("data/world.dat")     # 地标信息
data = read_data("data/sensor_data.dat")

INF = 9999

N = len(landmark)
observedLandmarks = np.repeat([False],N)

# init belief
mu = np.repeat(0,2*N+3)
robSigma = np.zeros([3,3])
robMapSigma = np.zeros([3,2*N])
mapSigma = np.eye(2*N) * INF
sigma = np.r_[np.c_[robSigma,robMapSigma],np.c_[robMapSigma.T,mapSigma]]

Flag_show_GUI = False

for t in range(len(data)):
    mu, sigma = prediction_step(mu,sigma, data[t]["Odometry"],t==0)
    mu, sigma, observedLandmarks = correction_step(mu,sigma,data[t]["Sensor"],observedLandmarks)
    print("Current state vector:")
    print(f"mu=\n{mu}")


print("Final system covariance matrix:"), print(sigma)
# Display the final state estimate
print("Final robot pose:")
print("mu_robot = ")
print(mu[:3])
print("sigma_robot = ")
print(sigma[:3,:3])

