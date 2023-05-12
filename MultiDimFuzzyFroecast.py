#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MultiDimFuzzyFroecast.py
# @Time      :2023/5/11 1:00 PM
# @Author    :Oliver

from torch.utils.data import DataLoader
from utils.Track_Generate import SNRNoise_Track_Dataset_Generate,CovarianceNoise_Track_Dataset_Generate
from PyRadarTrack.Model.TorchMovementModel import TorchMovementModelFactory
from FuzzyModel.MyModel import *
from FuzzyModel.Trainer import MSETrainer


if __name__ == '__main__':
    from utils.logger import rootlogger
    from torch.optim import lr_scheduler
    Simulate_time = 500
    dt = 0.1
    Sigma = 0.1

    # # region 运动模型相关
    # TchMMF = TorchMovementModelFactory()
    # CVModel = TchMMF.create('CVModel')(dt, Sigma)
    # CTModel = TchMMF.create('CTxyModel')(dt, Sigma, -0.35)
    # CAModel = TchMMF.create('CAModel')(dt, Sigma)
    # MovementModels = [CAModel, CTModel, CVModel]
    # # endregion
    # region --生成数据集
    batchSize = 2
    Win = 15
    # PP = PhysicalPred(MovementModels)
    # model = ClassifyFLS(PP,rule_num=64,TimeWin=Win).to(device=device)
    rootlogger('Train_NextStepModel')
    TFK1 = SNRNoise_Track_Dataset_Generate(Simulate_time, seed=None, xWin=Win)
    TFK2 = SNRNoise_Track_Dataset_Generate(Simulate_time, seed=None, xWin=Win)


    train_loader = DataLoader(dataset=TFK1,
                              batch_size=batchSize,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(dataset=TFK2,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    # endregion














