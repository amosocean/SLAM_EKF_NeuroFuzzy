#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MyModel.py
# @Time      :2023/4/25 9:22 PM
# @Author    :Oliver
import torch
import torch.nn as nn
from FuzzyModel.FLS import *
from FuzzyModel.Decorator import *
from .config import device

class BasicModel(torch.nn.Module):
    Ignore_name=["self",'__class__']
    def __init__(self, **kwargs):
        super().__init__()
        self._unfold(kwargs)
        for ignore in self.Ignore_name:
            if ignore in kwargs.keys():
                kwargs.pop(ignore)
        self.para_dict = kwargs

    def _unfold(self, dict, keyword="kwargs"):
        if keyword in dict:
            dict.update(dict.pop(keyword))
            self._unfold(dict, keyword)

    def quick_eval(self,*args):
        return self(torch.tensor(args,device=device))
    def forward(self,x):
        return x
    def get_init_para(self):
        for key in self.para_dict.keys():
            self.para_dict[key] = self.__getattribute__(key)
        return self.para_dict
    def _init_para_update(self,**kwargs):
        self.para_dict.update(kwargs)

class BasicOneStepModel(BasicModel):
    def __init__(self,xDim,rule_num,yDim=1,**kwargs):
        super().__init__(kwargs=dict(locals(),**kwargs))
        self.xDim = xDim
        self.rule_num = rule_num
        self.yDim = yDim
    # def quick_eval(self,*args):
    #     return self(torch.tensor(args,device=device))
    # def forward(self,x):
    #     return x

class BasicTimeSeriesModel(BasicModel):
    def __init__(self, xDim,xTimeDim,rule_num,yDim,yTimeDim,**kwargs):
        super().__init__(kwargs=dict(locals(),**kwargs))
        self.xDim = xDim
        self.xTimeDim=xTimeDim
        self.rule_num=rule_num
        self.yDim=yDim
        self.yTimeDim = yTimeDim
    # def quick_eval(self,*args):
    #     return self(torch.tensor(args,device=device))
    # def forward(self,x):
    #     return x

@scale()
class FLSLayer(BasicOneStepModel):
    def __init__(self,xDim,rule_num,yDim=1):
        super().__init__(xDim,rule_num,yDim=1)
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
class TSFLSLayer(BasicOneStepModel):
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
class TrapFLSLayer(BasicOneStepModel):
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
class StrictlyTrapFLSLayer(BasicOneStepModel):
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
class TwoHalfTrapFLSLayer(BasicOneStepModel):
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
        self.FLS_List=torch.nn.ModuleList()
        for i in range(xDim):
            self.FLS_List.append(FLSLayer(xTimeDim,rule_num))

    def forward(self,x):
        xs = torch.split(x,1,dim=-2)
        ys = []
        for i in range(self.xDim):
            ys.append(self.FLS_List[i](xs[i].squeeze(-2)))
        rtn = torch.stack(ys,dim=-2)
        return rtn

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
            torch.nn.BatchNorm1d(36),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=36,out_channels=72,kernel_size=yTimeDim,stride=1,padding=0),
            torch.nn.BatchNorm1d(72),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=72,out_channels=36,kernel_size=yTimeDim,stride=1,padding=0),
            torch.nn.BatchNorm1d(36),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=36,out_channels=yDim,kernel_size=yTimeDim,stride=1,padding=0),
            )

    def forward(self,x):
        # var, mean = (torch.var_mean(x, dim=-1))
        xs = torch.split(x,1,dim=-2)
        ys = []
        for i in range(self.xDim):
            ys.append(self.FLS_List[i](xs[i].squeeze(-2)))
        rtn = torch.stack(ys,dim=-2)
        rtn = self.dense2(rtn)
        return rtn

class Dense_AdoptTimeFLSLayer(BasicTimeSeriesModel):
    def __init__(self,xDim,xTimeDim,rule_num,yDim=1,yTimeDim=1):
        super().__init__(xDim,xTimeDim,rule_num,yDim,yTimeDim)
        self.NormPack = NormalizePacking(self.forward,xTimeDim)
        self.forward = self.NormPack.forward
        # self.AlterNorm = Norm_layer(yTimeDim)
        self.FLS_List=torch.nn.ModuleList()
        for i in range(xDim):
            self.FLS_List.append(FLSLayer(xTimeDim,rule_num))
        #self.fc1=torch.nn.Linear(yDim,yDim)
        self.dense2=torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=yDim,out_channels=36,kernel_size=yTimeDim,stride=1,padding=0),
            torch.nn.BatchNorm1d(36),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=36,out_channels=72,kernel_size=yTimeDim,stride=1,padding=0),
            torch.nn.BatchNorm1d(72),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=72,out_channels=36,kernel_size=yTimeDim,stride=1,padding=0),
            torch.nn.BatchNorm1d(36),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=36,out_channels=yDim,kernel_size=yTimeDim,stride=1,padding=0),
            )

    def forward(self,x):
        pass

class BasicLSTMNet(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=4, output_size=5):
        super(BasicLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.act1=nn.LeakyReLU()
    def forward(self, x):
        x=x.permute(0, 2, 1)#调整为(N,Length,Input_channel)顺序
        # 初始化 LSTM 隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        # 前向传递 LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # 使用线性层来将 LSTM 输出映射到输出空间
        out = self.fc(out[:, -1, :])
        out = self.act1(out)
        #return out.squeeze(dim=-1)
        return out.unsqueeze(dim=-1)
class LSTMNet(BasicLSTMNet):
    def __init__(self, xDim,xTimeDim,hidden_size,num_layers=1,yDim=1,yTimeDim=1):#yTimeDim=1 是out = self.fc(out[:, -1, :])决定的，还没有完成动态设计
        assert yTimeDim==1,"还不支持不是 yTimeDim==1的情况"
        super(LSTMNet,self).__init__(input_size=xDim, hidden_size=hidden_size, num_layers=num_layers, output_size=yDim)
        self.NormPack = NormalizePacking(self.forward,xTimeDim,channel_num=9)
        self.forward = self.NormPack.forward




