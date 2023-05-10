#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MyNextStepModel.py
# @Time      :2023/5/10 1:29 PM
# @Author    :Oliver
import torch

from FuzzyModel.FLS import *
from FuzzyModel.Decorator import *
from config import device

from utils.Track_Generate import Random_Track_Dataset_Generate
from PyRadarTrack.Model.TorchMovementModel import TorchMovementModelFactory
from FuzzyModel.MyModel import *
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicalPred(torch.nn.Module):
    def __init__(self,MovementModels):
        super().__init__()
        self.MovementModels = torch.nn.ModuleList(MovementModels)


    def forward(self,x):
        """
        [...,x] -> [...,modelNum,x]
        """
        PredX = []
        for Model in self.MovementModels:
            PredX.append(Model(x))
        return torch.stack(PredX,dim=-2)

class ClassifyFLS(torch.nn.Module):
    def __init__(self,Physical_pred:PhysicalPred,xDim=9,rule_num=16, TimeWin=5):
        super().__init__()
        self.xDim=xDim
        self.PPred = Physical_pred
        PModel_Num = len(self.PPred.MovementModels)

        self.Norm = torch.nn.LayerNorm(TimeWin)
        # self.FLS = FLSLayer(xDim,rule_num,PModel_Num)
        self.FLS_List=torch.nn.ModuleList()
        for i in range(xDim):
            self.FLS_List.append(FLSLayer(TimeWin, rule_num,PModel_Num))

        self.FLS2 = FLSLayer(xDim,rule_num)
        self.softmax = torch.nn.Softmax(dim = -2)

    def forward(self,x):
        norm_x = self.Norm(x)
        PM_x_pre = self.PPred(x[..., -1])
        # xs = torch.split(norm_x,1,dim=-2)
        ys = []
        for i in range(self.xDim):
            ys.append(self.FLS_List[i](norm_x[:,i,:]))
        rtn = torch.stack(ys,dim=-1)
        classify = self.FLS2(rtn)

        return torch.sum(PM_x_pre * self.softmax(classify),dim=-2)


EazyTest = True
if __name__ == '__main__':
    Simulate_time = 500
    dt = 0.1
    Sigma = 0.1
    TFK1 = Random_Track_Dataset_Generate(Simulate_time, seed=None)
    TFK2 = Random_Track_Dataset_Generate(Simulate_time, seed=None)
    # region 运动模型相关
    TchMMF = TorchMovementModelFactory()
    CVModel = TchMMF.create('CVModel')(dt, Sigma)
    CTModel = TchMMF.create('CTxyModel')(dt, Sigma, -0.35)
    CAModel = TchMMF.create('CAModel')(dt, Sigma)
    MovementModels = [CAModel, CTModel, CVModel]
    # endregion
    benchSize = 4
    Win = 5
    PP = PhysicalPred(MovementModels)
    model = ClassifyFLS(PP,rule_num=64,TimeWin=Win)
    # region 简单测测输入输出
    if EazyTest:
        X0 = torch.tensor([3000,10,0.1]*3) * torch.randn(benchSize,9)
        X1PhysicPred = PP(X0)
        X1FLSPred = model(torch.stack([X0]*Win,-1))
    # endregion





    # Models = TFK1.MovementModels





