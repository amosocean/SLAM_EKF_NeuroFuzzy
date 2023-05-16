#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MismatchIMM.py
# @Time      :2023/5/13 5:13 PM
# @Author    :Oliver
import torch

from FuzzyModel import *
from PyRadarTrack import *
from PyRadarTrack.Model.TorchMovementModel import TorchMovementModelFactory
from utils.Track_Generate import *
from utils.logger import *
from torch.optim import lr_scheduler
class CustomFLS(FLSLayer):
    pass

class TotalModel(torch.nn.Module):
    def __init__(self,xDim=9,xWin=5,rule_num=64,ModelList=None,Conv_Weight=None):
        super().__init__()
        self.xWin = xWin
        self.ModelList = torch.nn.ModuleList(ModelList)
        self.Conv_Wight = torch.ones([xDim]) if Conv_Weight is None else Conv_Weight
        self.FLS = CustomFLS(xWin,rule_num,1)   # 同构FLS

    def ModelForecast(self,x):
        Tmp = []
        for Model in self.ModelList:
            Tmp.append(Model(x))
        Fzs = torch.stack(Tmp,-3)
        return Fzs
    def ModelScore(self,Zs):
        assert Zs.shape[-2] == self.xWin+1, "需要从k-t~k共t+1个数据"
        Fzs = self.ModelForecast(Zs)
        Diff = Fzs[...,:-1,:] - Zs[...,1:,:].unsqueeze(-3)
        rtn = torch.sqrt(torch.sum((self.Conv_Wight * Diff / self.Conv_Wight.sum(-1)) ** 2, dim=-1))
        return (-torch.log(rtn)).softmax(-2)

    def appendModel(self, movement_model):
        self.ModelList.append(movement_model)
    def forward(self, x_last, Zs):
        Score = self.ModelScore(Zs.transpose(-1,-2))
        matchScore = self.FLS(Score).softmax(-2).unsqueeze(-1)
        x_now_pre = self.ModelForecast(x_last.unsqueeze(-2))    # 这里也许意味着有多输入一段x_last的潜力？

        return torch.sum(x_now_pre*matchScore,dim=-3)


class CustomDataset(CovarianceNoise_Track_Dataset_Generate):
    def __getitem__(self, idx):
        k = idx+self.xWin
        # pure_x = self.pure_track[:,k-1]
        pure_x = self.noisy_track[:,k-1]

        z_sample = self.noisy_track[:, idx: idx + self.xWin + 1]
        pure_next = self.pure_track[:, k: k + self.yWin]
        return pure_x, z_sample, pure_next

class CustomMSETrainer(MSETrainer):
    def workStep(self, loader, ifRecord=True, ifEval=False):
        total_loss = 0
        for batch in loader:
            real_x_last =  batch[0].to(self.device)
            sample = batch[1].to(self.device)
            gts = batch[2].to(self.device).transpose(-1,-2)   #转移到gpu
            batch_len = sample.shape[0]
            pred = self.model(real_x_last,sample)
            loss = torch.nn.functional.mse_loss(pred, gts)
            if not ifEval:
                self.optimizer.zero_grad()  # pytorch会积累梯度，在优化每个batch的权重的梯度之前将之前计算出的每个权重的梯度置0
                loss.backward()  # 在最后一个张量上调用反向传播方法，在计算图中计算权重的梯度
                self.optimizer.step()  # 使用预先设置的学习率等参数根据当前梯度对权重进行更新
            total_loss += loss * batch_len
            if ifRecord:
                self.tmp_save_data_pred.extend(pred.tolist())
                self.tmp_save_data_real.extend(gts.tolist())
        loss = total_loss / len(loader.dataset)

        return loss

class tracker():
    def __init__(self,x0,Model):
        self.X = x0
        self.model=Model

    def step(self,z):
        pass



if __name__ == '__main__':
    Cov = torch.diag(torch.tensor([1e1,1e-2,1e-4]*3))
    rootlogger('Train_MismatchModel')
    Simulate_time = 500
    dt = 0.1
    Sigma = 0.1
    Win = 5
    # region 准备训练用数据
    TFK1 = CustomDataset(Simulate_time, seed=None, xWin=Win).add_noise(Cov*1)
    TFK2 = CustomDataset(Simulate_time, seed=None, xWin=Win).add_noise(Cov*1)

    train_loader = DataLoader(dataset=TFK1, batch_size=2,shuffle=True,pin_memory=True)
    test_loader = DataLoader(dataset=TFK2, batch_size=1,shuffle=False,pin_memory=True)
    # endregion

    # region 硬定义运动模型
    TchMMF = TorchMovementModelFactory()
    CVModel = TchMMF.create('CVModel')(dt, Sigma)
    CTModel = TchMMF.create('CTxyModel')(dt, Sigma, -0.35)
    CAModel = TchMMF.create('CAModel')(dt, Sigma)
    MovementModels = [CAModel, CTModel, CVModel]
    # endregion

    model = TotalModel(ModelList=MovementModels)

    # region 训练模型
    print(model.parameters)
    epoch_num = 10
    learning_rate = 1e-1
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,200,400,600], gamma=0.5)

    Train = CustomMSETrainer(model=model, loader_train=train_loader, loader_test=test_loader, optimizer=optimizer,
                       lrScheduler=scheduler,logName='Train_NextStepModel')

    train_loss, test_loss = Train.run(epoch_num, div=2, show_loss=True)

    # endregion

    # region 仿真
    Est=[]
    X = X0 = TFK2.get_pure_track()[:,Win]
    zs = TFK2.get_noisy_track()
    for t in range(Win,Simulate_time):
        X_pre = model(X, zs[:,t-Win:t+1])[0]
        X = X_pre
        Est.append(X)

    TensorEst = torch.stack(Est).transpose(0,1)
    # endregion

    # region 绘图
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    def draw_3D(Ax, data_draw, label):
        data_draw = np.array(data_draw)
        Ax.plot3D(*data_draw, label=label)

    data_draw_1 = np.array(TFK2.get_pure_track()[[0,3,6]].detach())
    data_draw_2 = np.array(TFK2.get_noisy_track()[[0,3,6]].detach())
    data_draw_3 = np.array(TensorEst[[0,3,6]].detach())
    draw_3D(ax,data_draw_1,"True")
    draw_3D(ax,data_draw_2,"Measure")
    draw_3D(ax,data_draw_3,"Est")

    plt.legend()
    plt.show()

    # fig2 = plt.figure()
    # x = torch.arange(Simulate_time)*dt
    # plt.subplot(2,2,1)
    # plt.plot(x, data_draw_2[0],label="Measure")
    # plt.plot(x[Win-1:], data_draw_3[ 0],label="Est")
    # plt.plot(x, data_draw_1[0], label="True")
    # plt.legend()
    #
    # plt.subplot(2,2,2)
    # plt.plot(x, data_draw_2[1],label="Measure")
    # plt.plot(x[Win-1:], data_draw_3[ 1],label="Est")
    # plt.plot(x, data_draw_1[1], label="True")
    # plt.legend()
    #
    # plt.subplot(2,2,3)
    # plt.plot(x, data_draw_2[2],label="Measure")
    # plt.plot(x[Win-1:], data_draw_3[ 2],label="Est")
    # plt.plot(x, data_draw_1[2], label="True")
    # plt.legend()

    # plt.show()

    #endregion













