#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyPlotSupport.py
# @Time      :2023/4/26 3:56 PM
# @Author    :Oliver

from matplotlib import pyplot as plt
import os
import torch
from FuzzyModel.FLSMF import GaussianFunction,TrapFunction
import numpy as np
Filename_front = ""
Save_dir_root = "Simulation"
Save_dir_leaf = "Singleton_default_max-min_min_height"
color_list = ["r","g","b","y","o"]

def genPath(filename):
    return os.path.join(Save_dir_root, Save_dir_leaf, "{}{}".format(Filename_front, filename))

def Parameter_show_Gaussian(xDim,ruleDim, Para_mean,Para_Sigma):
    sample_num = 100
    fig = plt.figure(figsize=[5, 5])
    # ax = plt.axes((0.1, 0.55, 0.8, 0.4), projection="3d")
    ax = plt.axes(projection="3d")
    sample = torch.linspace(0, 1, sample_num)
    xs = sample[:,None,None]
    GF = GaussianFunction([xDim,ruleDim], Para_mean,Para_Sigma,
                          True,True)
    draw_data = GF(xs).detach()     # [Sample, xDim, Rule]
    # draw_x = xs.repeat(1,xDim,ruleDim)
    for rule in range(ruleDim):
        for x_idx in range(xDim):
            ax.plot(sample, [rule+0.01*x_idx]*sample_num, draw_data[:,x_idx,rule],c=color_list[x_idx],linewidth=1)

    ax.set_xlabel('x')
    ax.set_ylabel('rule_index')
    ax.set_zlabel('Mu')
    ax.set_zlim([0, 1])
    # ax.view_init(elev=-140, azim=35)
    # ax.set_title('(a)', y=-0.13)
    # plt.savefig(f"x1={x1[0]}.png")
    # plt.show()

    # plt.figure(figsize=[5, 5])
    # ax = plt.axes((0.1, 0.05, 0.8, 0.4), projection="3d")
    # ax.plot_surface(X3, X2, Z2, cmap='viridis')
    # ax.set_xlabel('x3(Movement)')
    # ax.set_ylabel('x2(Battery)')
    # ax.set_zlabel('Mu')
    # ax.set_zlim([10, 0])
    # ax.view_init(elev=-140, azim=35)
    # ax.set_title('(b)', y=-0.13)
    # plt.savefig(genPath(f"upper:x1={x1[0]},bottem:x1={x1[1]}.png"), transparent=True)
    plt.show()
    return fig

def draw_loss(train_loss,test_loss):
    fig = plt.figure()
    train_x = np.array([i for i,j in train_loss.items()])
    train_y = np.array([j.detach() for i,j in train_loss.items()])
    test_x = np.array([i for i,j in test_loss.items()])
    test_y = np.array([j.detach() for i,j in test_loss.items()])
    plt.plot(train_x,train_y,label="Train")
    plt.plot(test_x,test_y,label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss(RMSE)")
    plt.legend()
    # plt.savefig("output/Fuzzy_loss.png")
    # plt.show()
    # fig.show()
    return fig




