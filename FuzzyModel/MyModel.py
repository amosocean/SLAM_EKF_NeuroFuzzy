#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MyModel.py
# @Time      :2023/4/25 9:22 PM
# @Author    :Oliver
import torch
from FuzzyModel.FLS import *
from FuzzyModel.Decorator import *
from .config import device
class BasicModel(torch.nn.Module):
    def __init__(self,xDim,rule_num,yDim=1):
        super().__init__()
        self.xDim=xDim
        self.rule_num = rule_num
        self.yDim=yDim
    def quick_eval(self,*args):
        return self(torch.tensor(args,device=device))
    def forward(self,x):
        return x

class BasicTimeSeriesModel(torch.nn.Module):
    def __init__(self, xDim,xTimeDim,rule_num,yDim,yTimeDim):
        super().__init__()
        self.xDim = xDim
        self.xTimeDim=xTimeDim
        self.rule_num=rule_num
        self.yDim=yDim
        self.yTimeDim = yTimeDim
    def quick_eval(self,*args):
        return self(torch.tensor(args,device=device))
    def forward(self,x):
        return x

@scale()
class FLSLayer(BasicModel):
    def __init__(self,xDim,rule_num,yDim=1):
        super().__init__(xDim,rule_num)
        self.Fuzzifier = FuzzifierLayer(xDim)
        self.Inference = GaussianInferenceLayer(xDim,rule_num)
        self.Defuzzifier = HeightDefuzzifierLayer(rule_num,yDim)
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

@scale()
class StrictlyTrapFLSLayer(BasicModel):
    def __init__(self,xDim,rule_num):
        super().__init__(xDim,rule_num)
        self.Fuzzifier = FuzzifierLayer(xDim)
        self.Inference = StrictlyTrapInferenceLayer(xDim,rule_num)
        self.Defuzzifier = HeightDefuzzifierLayer(rule_num)
        # self.Defuzzifier = DefuzzifierLayer(rule_num,ySampleDim=100)

    def forward(self,input):
        input = self.Fuzzifier(input)
        input = self.Inference(input)
        input = self.Defuzzifier(input)
        return input

@scale()
class TwoHalfTrapFLSLayer(BasicModel):
    def __init__(self,xDim,rule_num):
        super().__init__(xDim,rule_num)
        self.Fuzzifier = FuzzifierLayer(xDim)
        self.Inference = HalfTrapInferenceLayer(xDim, rule_num)
        self.Inference2 = HalfTrapInferenceLayer(xDim, rule_num)
        self.Defuzzifier = HeightDefuzzifierLayer(rule_num)
        # self.Defuzzifier = DefuzzifierLayer(rule_num,ySampleDim=100)

    def forward(self, input):
        input = self.Fuzzifier(input)
        input = self.Inference(input)
        input = self.Inference2(input)
        input = self.Defuzzifier(input)
        return input

class AdoptTimeFLSLayer(BasicTimeSeriesModel):
    def __init__(self,xDim,xTimeDim,rule_num,yDim=1,yTimeDim=1):
        super().__init__(xDim,xTimeDim,rule_num,yDim,yTimeDim)
        self.Norm = FixNorm_layer(xTimeDim)
        # self.AlterNorm = Norm_layer(yTimeDim)
        self.FLS_List=torch.nn.ModuleList()
        for i in range(xDim):
            self.FLS_List.append(FLSLayer(xTimeDim,rule_num))

    def forward(self,x):
        x_norm, mean,var = self.Norm(x)
        # var, mean = (torch.var_mean(x, dim=-1))
        xs = torch.split(x_norm,1,dim=-2)
        ys = []
        for i in range(self.xDim):
            ys.append(self.FLS_List[i](xs[i].squeeze(-2)))
        rtn = torch.stack(ys,dim=-2)

        return rtn * var + mean

class PackingAdoptTimeFLSLayer(BasicTimeSeriesModel):
    def __init__(self,xDim,xTimeDim,rule_num,yDim=1,yTimeDim=1):
        super().__init__(xDim,xTimeDim,rule_num,yDim,yTimeDim)
        self.FLS_List=torch.nn.ModuleList()
        for i in range(xDim):
            self.FLS_List.append(FLSLayer(xTimeDim,rule_num))
        self.NormPack = NormalizePacking(self.forward,xTimeDim)
        self.forward = self.NormPack.forward

    def forward(self,x):
        # x_norm, mean,var = self.Norm(x)
        # var, mean = (torch.var_mean(x, dim=-1))
        xs = torch.split(x,1,dim=-2)
        ys = []
        for i in range(self.xDim):
            ys.append(self.FLS_List[i](xs[i].squeeze(-2)))
        rtn = torch.stack(ys,dim=-2)

        return rtn

class AdoptTimeFLSLayer_Dense(BasicTimeSeriesModel):
    def __init__(self,xDim,xTimeDim,rule_num,yDim=1,yTimeDim=1):
        super().__init__(xDim,xTimeDim,rule_num,yDim,yTimeDim)
        self.Norm = FixNorm_layer(xTimeDim)
        # self.AlterNorm = Norm_layer(yTimeDim)
        self.FLS_List=torch.nn.ModuleList()
        for i in range(xDim):
            self.FLS_List.append(FLSLayer(xTimeDim,rule_num))
        #self.fc1=torch.nn.Linear(yDim,yDim)
        self.dense2=torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=yDim,out_channels=36,kernel_size=yTimeDim,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=36,out_channels=yDim,kernel_size=yTimeDim,stride=1,padding=0)
            )

    def forward(self,x):
        x_norm, mean,var = self.Norm(x)
        # var, mean = (torch.var_mean(x, dim=-1))
        xs = torch.split(x_norm,1,dim=-2)
        ys = []
        for i in range(self.xDim):
            ys.append(self.FLS_List[i](xs[i].squeeze(-2)))
        rtn = torch.stack(ys,dim=-2)
        rtn = self.dense2(rtn)
        return rtn * var + mean

class Dense_AdoptTimeFLSLayer(BasicTimeSeriesModel):
    def __init__(self,xDim,xTimeDim,rule_num,yDim=1,yTimeDim=1):
        super().__init__(xDim,xTimeDim,rule_num,yDim,yTimeDim)
        self.Norm = FixNorm_layer(xTimeDim)
        # self.AlterNorm = Norm_layer(yTimeDim)
        self.FLS_List=torch.nn.ModuleList()
        for i in range(xDim):
            self.FLS_List.append(FLSLayer(xTimeDim,rule_num))
        #self.fc1=torch.nn.Linear(yDim,yDim)
        self.dense2=torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=yDim,out_channels=yDim,kernel_size=yTimeDim,stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=yDim,out_channels=yDim,kernel_size=yTimeDim,stride=1,padding=0)
            )

    def forward(self,x):
        x_norm, mean,var = self.Norm(x)
        # var, mean = (torch.var_mean(x, dim=-1))
        xs = torch.split(x_norm,1,dim=-2)
        ys = []
        for i in range(self.xDim):
            ys.append(self.FLS_List[i](xs[i].squeeze(-2)))
        rtn = torch.stack(ys,dim=-2)
        rtn = self.dense2(rtn)
        return rtn * var + mean

class DifferenceLayer(BasicTimeSeriesModel):
    def __init__(self, xDim, xTimeDim, rule_num, yDim=1, yTimeDim=1):
        super().__init__(xDim, xTimeDim, rule_num, yDim, yTimeDim)

    def forward(self, x):
        pass


