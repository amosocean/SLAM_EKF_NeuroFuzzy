#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MyNextStepModel.py
# @Time      :2023/5/10 1:29 PM
# @Author    :Oliver
import torch

from FuzzyModel.FLS import *
from FuzzyModel.Decorator import *
from config import device

from torch.utils.data import DataLoader
from utils.Track_Generate import Random_Track_Dataset_Generate
from PyRadarTrack.Model.TorchMovementModel import TorchMovementModelFactory
from FuzzyModel.MyModel import *
from FuzzyModel.Trainer import MSETrainer

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
    def __init__(self,Physical_pred:PhysicalPred,xDim=9, rule_num=16, TimeWin=5):
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
            ys.append(self.FLS_List[i](norm_x[...,i,:]))
        rtn = torch.stack(ys,dim=-1)
        classify = self.softmax(self.FLS2(rtn))

        return torch.sum(PM_x_pre * classify,dim=-2).unsqueeze(-1)

class SlidingWindow(object):
    def __init__(self,EvalModel,xDim=9, yDim=9, TimeWin=5):
        super().__init__()
        self.EvalModel=EvalModel
        self.TimeWin=5
        self.dataTemp = torch.empty([xDim,0])
        self.Est = torch.empty([yDim,0])


    def __call__(self, x):
        self.dataTemp = torch.concat([self.dataTemp,x],dim=1)
        if self.dataTemp.shape[1]>=self.TimeWin:
            Est = self.EvalModel(self.dataTemp[:, -self.TimeWin:])
            self.Est = torch.concat([self.Est,Est],dim=1)
            return Est
        else:
            return None

if __name__ == '__main__':
    # import numpy as np
    from utils.logger import rootlogger
    from torch.optim import lr_scheduler
    Simulate_time = 500
    dt = 0.1
    Sigma = 0.1
    # region 运动模型相关
    TchMMF = TorchMovementModelFactory()
    CVModel = TchMMF.create('CVModel')(dt, Sigma)
    CTModel = TchMMF.create('CTxyModel')(dt, Sigma, -0.35)
    CAModel = TchMMF.create('CAModel')(dt, Sigma)
    MovementModels = [CAModel, CTModel, CVModel]
    # endregion
    batchSize = 2
    Win = 5
    PP = PhysicalPred(MovementModels)
    model = ClassifyFLS(PP,rule_num=64,TimeWin=Win).to(device=device)
    rootlogger('Train_NextStepModel')
    # region #简单测测输入输出
    # if True:
    #     X0 = torch.tensor([3000,10,0.1]*3) * torch.randn(batchSize,9)
    #     X1PhysicPred = PP(X0)
    #     X1FLSPred = model(torch.stack([X0]*Win,-1))
    # endregion

    # region 设计数据集
    # TFK1 = Random_Track_Dataset_Generate(Simulate_time, seed=None)
    # TFK2 = Random_Track_Dataset_Generate(Simulate_time, seed=None)
    TFK1 = Random_Track_Dataset_Generate(Simulate_time,seed=None,xWin=Win)
    TFK2 = Random_Track_Dataset_Generate(Simulate_time,seed=None,xWin=Win)
    # X0 = np.array([3300, 2, 1e-3, 3400, 3, 3e-3, 3500, 4, 4e-4])
    # X1 = np.array([3300, -2, -1e-3, 3400, -3, -3e-3, 3500, -4, -4e-4])
    #
    # TFK1.gen_randomTrack(X0)
    # TFK2.gen_randomTrack(X1)

    TFK1_noise=TFK1.add_noise(snr=-0)
    TFK2_noise=TFK2.add_noise(snr=-0)

    train_loader = DataLoader(dataset=TFK1_noise,
                              batch_size=batchSize,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(dataset=TFK2_noise,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    # endregion

    # region 训练模型
    print(model.parameters)
    epoch_num = 10
    learning_rate = 1e-1
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,200,400,600], gamma=0.5)

    Train = MSETrainer(model=model, loader_train=train_loader, loader_test=test_loader, optimizer=optimizer,
                       lrScheduler=scheduler,logName='Train_NextStepModel')

    train_loss, test_loss = Train.run(epoch_num, div=2, show_loss=True)

    # endregion

    # region 效果展示
    SW = SlidingWindow(model)
    Est = None
    for b in TFK2.TrackData_noisy:
        if Est is not None:
            loss = torch.nn.functional.mse_loss(Est, b.unsqueeze(-1))
            print(loss)
        Est = SW(b.unsqueeze(-1))



    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np

    fig = plt.figure()

    ax = plt.axes(projection='3d')
    def draw_3D(Ax, data_draw, label):
        data_draw = np.array(data_draw)
        x = data_draw[:, 0]
        y = data_draw[:, 1]
        z = data_draw[:, 2]
        Ax.plot3D(x, y, z, label=label)

    data_draw_1 = TFK2.TrackData[:,[0,3,6]].detach()
    data_draw_2 = TFK2.TrackData_noisy[:,[0,3,6]].detach()
    data_draw_3 = SW.Est[[0,3,6],:].T.detach()
    draw_3D(ax,data_draw_1,"True")
    draw_3D(ax,data_draw_2,"Measure")
    draw_3D(ax,data_draw_3,"Est")

    plt.legend()
    plt.show()

    # endregion


