#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MyModel.py
# @Time      :2023/4/25 9:22 PM
# @Author    :Oliver
import torch
from FuzzyModel.FLS import *
from utils.Decorator import *

class BasicModel(torch.nn.Module):
    def __init__(self,xDim,rule_num,yDim=1):
        super().__init__()
        self.xDim=xDim
        self.rule_num = rule_num
        self.yDim=yDim
    def forward(self,input):
        return input

@scale()
class FLSLayer(BasicModel):
    def __init__(self,xDim,rule_num):
        super().__init__(xDim,rule_num)
        self.Fuzzifier = FuzzifierLayer(xDim)
        self.Inference = GaussianInferenceLayer(xDim,rule_num)
        self.Defuzzifier = HeightDefuzzifierLayer(rule_num)
        # self.Defuzzifier = DefuzzifierLayer(rule_num,ySampleDim=100)

    def forward(self,input):
        input = self.Fuzzifier(input)
        input = self.Inference(input)
        input = self.Defuzzifier(input)
        return input

@scale()
class TSFLSLayer(BasicModel):
    def __init__(self,xDim,rule_num):
        super().__init__(xDim,rule_num)
        self.Fuzzifier = FuzzifierLayer(xDim)
        self.Inference = GaussianInferenceLayer(xDim,rule_num)
        self.Defuzzifier = TSDefuzzifierLayer(xDim,rule_num)
        # self.Defuzzifier = DefuzzifierLayer(rule_num,ySampleDim=100)

    def forward(self,input):
        input = self.Fuzzifier(input)
        input = self.Inference(input)
        input = self.Defuzzifier(input)
        return input

@scale()
class TrapFLSLayer(BasicModel):
    def __init__(self,xDim,rule_num):
        super().__init__(xDim,rule_num)
        self.Fuzzifier = FuzzifierLayer(xDim)
        self.Inference = TrapInferenceLayer(xDim,rule_num)
        self.Defuzzifier = HeightDefuzzifierLayer(rule_num)
        # self.Defuzzifier = DefuzzifierLayer(rule_num,ySampleDim=100)

    def forward(self,input):
        input = self.Fuzzifier(input)
        input = self.Inference(input)
        input = self.Defuzzifier(input)
        return input