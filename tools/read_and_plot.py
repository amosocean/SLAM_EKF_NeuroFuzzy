#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :read_and_plot.py
# @Time      :2023/4/14 8:39 PM
# @Author    :Kinddle
import matplotlib.pyplot as plt
import numpy as np

from tools import *
import gif

def read_world(filename):
    rtn = pd.DataFrame(columns=["id", "x", "y"])
    # rtn_dt = np.dtype({'names': ['id','x','y'],
    #                     'formats': [int, float,float]})

    with open(filename, "r") as F:
        line = " "
        while line:
            line = F.readline()
            if line:
                id, x, y = line.split(None)

                rtn = rtn.append({"id": int(id)-1,
                                  "x": float(x),
                                  "y": float(y)}, ignore_index=True)
    return rtn

def read_data(filename):
    rtn = []
    basic_dict = {"Odometry": None, "Sensor": []}
    with open(filename, "r") as F:
        line = " "
        while line:
            line = F.readline()
            if line:
                Type, p1, p2, p3 = line.split(None)
                if Type == "ODOMETRY":
                    rtn.append(basic_dict.copy())
                    basic_dict = {"Odometry": None, "Sensor": []}
                    basic_dict["Odometry"] = {"r1":eval(p1),"t":eval(p2),"r2":eval(p3)}
                elif Type == "SENSOR":
                    basic_dict["Sensor"].append({"id":eval(p1)-1,"range":eval(p2),"bearing":eval(p3)})
                else:
                    raise IOError("未知的数据类型")
    return rtn[1:]
@gif.frame
def plot_state(mu, sigma,landmarks,timestep, observedLandmarks,z,window):
    """
    Visualizes the state of EKF SLAM algorithm

    the plot including the following information:
    1. map ground truth (black '+')
    2. current robot pose estimate(red)
    3. current landmark pose estimate(blue)
    4. visualization of the observations made at this time step (black line between robot and landmark)
    """
    plt.figure()
    plt.xlim([-2, 12])
    plt.ylim([-2, 12])
    plt.scatter(landmarks.x,landmarks.y,c="black", marker='P',s=24)
    drawProbEllipse(mu[:2],sigma[:2,:2],5.991,"red")

    for i in range(len(observedLandmarks)):
        if observedLandmarks[i]:
            xid = 2*i+3
            yid = 2*i+4
            plt.scatter(mu[xid],mu[yid],c="blue",marker="o",s=24)
            drawProbEllipse(mu[xid:yid+1], sigma[xid:yid+1,xid:yid+1],5.991,"blue")

    for z_iter in z:
        sX = mu[2*z_iter["id"]+3]
        sY = mu[2*z_iter["id"]+4]
        plt.plot([mu[0],sX],[mu[1],sY], c="black", linewidth=2)

    drawrobot(mu[:3], 'g', 3, 0.3, 0.3)

def drawrobot(mu,color,type=2,B=0.4,L=0.6):
    """

    :param mu:
    :param color:
    :param type:
    :param B:
    :param L:
    :return:
    """

    WT = 0.03
    WD = 0.2
    RR = WT / 2
    RRR = 0.04
    HL = 0.09
    CS = 0.1
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]
    T = np.array([x,y])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    if type==3:
        radius = (B + WT) / 2
        h1 = drawProbEllipse(mu, np.eye(2), radius**2, color)
        p = R.dot(np.array([1,0])) * radius + T
        h2 = plt.plot([T[0], p[0]], [T[1], p[1]], c=color, linewidth=2)
        h = [h1,h2]

    return h


def drawProbEllipse(x,C,s,color):
    """
    < Confidence Ellipse >
    using Unscented transform
    ~ url: https://blog.csdn.net/qq_36097393/article/details/87605173
    :param x: center of Ellipse using [:2]
    :param C: Cov. using [:2,:2]
    :param s: s is related to the confidence level.
              A 95% confidence level corresponds to s = 5.991.
    """
    N_POINTS = 100
    u = np.linspace(0,np.pi*2,N_POINTS)
    if np.all(C[:2,:2]==0):
        L_sqrt_C = C[:2,:2]
    else:
        L_sqrt_C = np.linalg.cholesky(C[:2,:2])
    p = np.c_[np.cos(u), np.sin(u)].T * np.sqrt(s)
    p = L_sqrt_C.dot(p) + x[:2,None]
    return plt.plot(p[0,:],p[1,:],c=color,linewidth=2)



#
# def drawEllipse(x, a, b, color):
#     N_POINTS = 100
#     u = np.linspace(0,np.pi*2,N_POINTS)
#     p = np.c_[a*np.cos(u), b*np.sin(u)].T
#     # p_y =
#
#     # Translate and rotate
#     angle = x[2]
#     R = np.array([[np.cos(angle),-np.sin(angle)],
#                   [np.sin(angle),np.cos(angle)]])
#     T = x[[0,1],None]
#     p = R.dot(p) + T
#
#     return plt.plot(p[0,:],p[1,:],c=color,linewidth=2)
#
#
#
# def old1_drawProbEllipse(x, C, s, color):
#     """
#     详见 Confidence Ellipse
#     ~ url: https://blog.csdn.net/qq_36097393/article/details/87605173
#     :param x: center of Ellipse
#     :param C: Cov.
#     :param s: s is related to the confidence level.
#               A 95% confidence level corresponds to s = 5.991.
#     """
#     N_POINTS = 100
#     u = np.linspace(0,np.pi*2,N_POINTS)
#     eig_v, eig_vec = np.linalg.eig(C[:2,:2])
#     a, b = np.sqrt(eig_v*s)
#     p = np.c_[a*np.cos(u), b*np.sin(u)].T
#
#     # Translate and rotate
#     R = eig_vec
#     T = x[[0,1],None]
#     p = R.dot(p) + T
#
#     return plt.plot(p[0,:],p[1,:],c=color,linewidth=2)
#
#
#
#
#
# def old_drawProbEllipse(x,C,s,color):
#     """
#     详见 Confidence Ellipse
#     ~ url: https://blog.csdn.net/qq_36097393/article/details/87605173
#     :param x: center of Ellipse
#     :param C: Cov.
#     :param s: s is related to the confidence level.
#               A 95% confidence level corresponds to s = 5.991.
#     """
#     eig_v, eig_vec = np.linalg.eig(C[:2,:2])
#     a, b = np.sqrt(eig_v)
#     s_sqrt = np.sqrt(s)
#     a, b =a*s_sqrt, b*s_sqrt
#     a_vec = eig_vec[:, 0]
#     angle = np.arctan2(a_vec[1],a_vec[0])
#     return drawEllipse(np.r_[x[:2],angle],a,b,color)
#
