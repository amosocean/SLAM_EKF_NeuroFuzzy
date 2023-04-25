#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model.py
# @Time      :2023/4/24 4:55 PM
# @Author    :Oliver
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .FuzzyMembershipFunction import GaussianMF
from .FuzzyLogicSystem import *

class FuzzyLayer(nn.Module):
    def __init__(self,xDim,rule_num,x_offset=0,x_scale=1,y_offset=0,y_scale=1):
        super().__init__()
        self.Fuzzifier = FuzzifierLayer(xDim)
        self.Inference = GaussianInferenceLayer(xDim,rule_num)
        self.Defuzzifier = HeightDefuzzifierLayer(rule_num)
        # self.Defuzzifier = DefuzzifierLayer(rule_num,ySampleDim=100)
        self.x_offset=x_offset
        self.y_offset = y_offset
        self.x_scale=x_scale
        self.y_scale=y_scale

    def forward(self,input):
        input = input*self.x_scale + self.x_offset
        input = self.Fuzzifier(input)
        input = self.Inference(input)
        input = self.Defuzzifier(input)
        return input * self.y_scale + self.y_offset

    def watch_rule(self,idx=None):
        if idx is None:
            Ant_mean = self.Inference.para_gauss_mean
            Ant_sigma = self.Inference.para_gauss_sigma
            con_mean = self.Defuzzifier.para_mean
            con_sigma = self.Defuzzifier.para_sigma
        else:
            Ant_mean = self.Inference.para_gauss_mean[:,0,idx]
            Ant_sigma = self.Inference.para_gauss_sigma[:,0,idx]
            con_mean = self.Defuzzifier.para_mean[:,0,idx]
            con_sigma = self.Defuzzifier.para_sigma[:,0,idx]

        return Ant_mean,Ant_sigma,con_mean,con_sigma

    def show_rule(self,idx):
        Am,As,Cm,Cs = self.watch_rule(idx)
        x = torch.linspace(0,1,100).unsqueeze(-1)
        F = GaussianMF(0,1)
        plt.subplot(2,1,1)
        plt.plot(x,F((x-Am)/As).detach().numpy())
        plt.legend(["x"+str(i) for i in range(Am.shape[0])])
        plt.subplot(2,1,2)
        plt.plot(x,F((x-Cm)/Cs).detach().numpy())
        plt.show()


