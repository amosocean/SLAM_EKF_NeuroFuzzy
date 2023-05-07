#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :IMMEKF_Final.py
# @Time      :2023/1/16 8:23 PM
# @Author    :Kinddle

import numpy as np

from PyRadarTrack import *
from PyRadarTrack.Model import *
from PyRadarTrack.Simulate import *
from PyRadarTrack.Model.FilterModel import IMMFilterModel, BasicEKFModel

if __name__ == '__main__':

    from FuzzyModel.FLS import FormalNorm_layer
    from FuzzyModel.MyModel import AdoptTimeFLSLayer,AdoptTimeFLSLayer_Dense
    import torch
    from torch.utils.data import DataLoader
    import torch.optim.lr_scheduler as lr_scheduler
    from utils.logger import rootlogger
    from FuzzyModel.Trainer import MSETrainer
    from utils.Track_Generate import Random_Track_Generate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Simulate_time = 500
    TFK1 = Random_Track_Generate(Simulate_time)
    TFK2 = Random_Track_Generate(Simulate_time)
    # region 规划初始点和初始速度
    X0 = np.array([3300, 2, 1e-3, 3400, 3, 3e-3, 3500, 4, 4e-4])
    X1 = np.array([3300, -2, -1e-3, 3400, -3, -3e-3, 3500, -4, -4e-4])
    TFK1.gen_randomTrack(X0)
    TFK2.gen_randomTrack(X1)
    # endregion

    batch_size = 5
    time_dim = 5
    Test = FormalNorm_layer([time_dim])
    train_loader = DataLoader(dataset=TFK1,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False)
    test_loader = DataLoader(dataset=TFK2,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=False)
    # A = Test(tensor_real_data[:time_dim])
    model = AdoptTimeFLSLayer_Dense(9, time_dim, 64, 9, 1).to(device=device)
    epoch_num = 10
    learning_rate = 10e-1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    rootlogger('Train')
    Train = MSETrainer(model=model, loader_train=train_loader, loader_test=test_loader, optimizer=optimizer,
                       lrScheduler=scheduler,logName='Train')

    train_loss, test_loss = Train.run(epoch_num, 2, True)

    Fuzzy_Est = []
    for b in test_loader:
        output = model(b[0]).squeeze()
        Fuzzy_Est.append(output)
    # for t in range(Simulate_time-time_dim):
    #     input = dataset_test[t][0]
    #     output = model(input).squeeze()
    #     Fuzzy_Est.append(output)

    Fuzzy_Est_tensor = torch.stack(Fuzzy_Est)

    # region [+]绘图
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    data_draw1 = TFK1.Track.get_real_data_all().iloc[:Simulate_time, [0, 3, 6]].to_numpy()
    data_draw3 = TFK2.Track.get_real_data_all().iloc[:Simulate_time, [0, 3, 6]].to_numpy()
    data_draw4 = np.array(Fuzzy_Est_tensor[:, [0, 3, 6]].detach().cpu())
    # data_draw2 = Xkf[[0, 3, 6], :].T
    # data_draw2 = recordsA["EstimateRecorder"].iloc[:, [0, 3, 6]].to_numpy()
    # data_draw3 = recordsB["EstimateRecorder"].iloc[:, [0, 3, 6]]

    ax = plt.axes(projection='3d')


    def draw_3D(Ax, data_draw, label):
        Ax.plot3D(data_draw[:, 0], data_draw[:, 1], data_draw[:, 2], label=label)


    # 三维线的数据
    # draw_3D(ax,data_draw1,"real")
    # draw_3D(ax,data_draw2,"Est")
    draw_3D(ax, data_draw3, "real2")
    draw_3D(ax, data_draw4, "FuzzyEst")

    plt.legend()
    plt.show()
    # endregion
