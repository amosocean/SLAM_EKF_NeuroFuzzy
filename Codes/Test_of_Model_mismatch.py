#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Test_of_Model_mismatch.py
# @Time      :2023/5/12 8:24 PM
# @Author    :Oliver
from PyRadarTrack.Model.TorchMovementModel import TorchMovementModelFactory
from utils.Track_Generate import CovarianceNoise_Track_Dataset_Generate
import torch

compress_weight = torch.ones([1,1,9])/9
def MatchScore(ModelList,Zs):
    Diff = []
    for Model in ModelList:
        Diff.append(Model(Zs))
    Fzs = torch.stack(Diff)
    Dif = Fzs[...,:-1,:] - Zs[...,1:,:]
    rtn = torch.sqrt(torch.sum((compress_weight * Dif / compress_weight.sum(-1)) ** 2, dim=-1))
    return (-torch.log(rtn)).softmax(-2)
    # return rtn/rtn.sum(0)

if __name__ == '__main__':
    from utils.logger import rootlogger
    from torch.optim import lr_scheduler
    Simulate_time = 500
    dt = 0.1
    Sigma = 0.1
    Win = 5
    TchMMF = TorchMovementModelFactory()
    CVModel = TchMMF.create('CVModel')(dt, Sigma)
    CTModel = TchMMF.create('CTxyModel')(dt, Sigma, -0.35)
    CAModel = TchMMF.create('CAModel')(dt, Sigma)
    MovementModels = [CAModel, CTModel, CVModel]

    TK_Cv = CovarianceNoise_Track_Dataset_Generate(Simulate_time,seed=600,xWin=Win,UsedModel=["CV"])
    TK_CT = CovarianceNoise_Track_Dataset_Generate(Simulate_time,seed=600,xWin=Win,UsedModel=["CT"])
    TK_CA = CovarianceNoise_Track_Dataset_Generate(Simulate_time,seed=600,xWin=Win,UsedModel=["CA"])
    TK_Mix = CovarianceNoise_Track_Dataset_Generate(Simulate_time,seed=600,xWin=Win)
    z = TK_Cv[0][0].T
    rtn = MatchScore([CVModel,CAModel,CTModel],z)


    cvFz = CVModel(z)
    caFz = CAModel(z)
    ctFz = CTModel(z)
    diffCv = cvFz[:-1] - z[1:]
    lossCv = cvFz[:, :-1] - z[:, 1:]
    lossCa = caFz[:, :-1] - z[:, 1:]
    lossCt = ctFz[:, :-1] - z[:, 1:]

    loss2Cv = torch.diag(1 / (z[:, :-1] + 1e-5).T @ lossCv)
    loss2Ca = torch.diag(1 / (z[:, :-1] + 1e-5).T @ lossCa)
    loss2Ct = torch.diag(1 / (z[:, :-1] + 1e-5).T @ lossCt)
    loss2Cv2 = ((1 / (z[:, :-1] + 1e-5).T).unsqueeze(-2) @ lossCv.T.unsqueeze(-1)).shape



    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np

    fig = plt.figure()
    ax = plt.axes(projection='3d')


    def draw_3D(Ax, data_draw, label):
        data_draw = np.array(data_draw)
        Ax.plot3D(*data_draw, label=label)


    data_draw_1 = np.array(TK_Cv.get_noisy_track()[[0, 3, 6]].detach())
    data_draw_2 = np.array(TK_CA.get_noisy_track()[[0, 3, 6]].detach())
    data_draw_3 = np.array(TK_CT.get_noisy_track()[[0, 3, 6]].detach())
    data_draw_4 = np.array(TK_Mix.get_noisy_track()[[0, 3, 6]].detach())
    # data_draw_2 = np.array(TFK2.get_noisy_track()[[0, 3, 6]].detach())
    # data_draw_3 = np.array(SW.Est[[0, 3, 6]].detach())
    draw_3D(ax, data_draw_1, "CV")
    draw_3D(ax, data_draw_2, "CA")
    draw_3D(ax, data_draw_3, "CT")
    draw_3D(ax, data_draw_4, "Mix")

    plt.legend()
    plt.show()


