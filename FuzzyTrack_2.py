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

# PyRadarTrack.

# if __name__ == '__main__':
#     dt = 0.1
#     Sigma = 0.01
#     Simulate_time = 500
#     # PyRadarTrack.Model.MovementModelFactory()
#     MMF = MovementModelFactory()
#     S_MF = SensorModelFactory()
#     SB = SimulationBox()
#     SB.SystemCfgUpdate({"Ts": 0.1,
#                         "QSigma": Sigma,
#                         "SimulationTimeTicks": Simulate_time})
#     # 生成轨迹
#     TFK1 = TargetFromKeyframe(SB)
#     TFK2 = TargetFromKeyframe(SB)
#     X0 = np.array([3300, 2, 1e-3, 3400, 3, 3e-3, 3500, 4, 4e-4])
#     X1 = np.array([3300, -2, -1e-3, 3400, -3, -3e-3, 3500, -4, -4e-4])
#     CVModel = MMF.create('CVModel')(dt, Sigma)
#     CTModel = MMF.create('CTxyModel')(dt, Sigma, -0.35)
#     CAModel = MMF.create('CAModel')(dt, Sigma)
#     TFK1.step(X0).run_Model(CAModel,200-1).run_Model(CTModel,250)\
#         .run_Model(CVModel,Simulate_time - 450 if Simulate_time>450 else 100)
#     TFK2.step(X1).run_Model(CAModel,200-1).run_Model(CTModel,250)\
#         .run_Model(CVModel,Simulate_time - 450 if Simulate_time>450 else 100)
#
#     # 生成传感器，并配置
#     # Sensor = S_MF.create("Radar_B")([0, 0, 0] * 3, parents=SB)
#     # Sensor = S_MF.create("Radar_B")([1000, 1000, 1000] * 3, parents=SB)
#     Sensor = S_MF.create("Radar_B")([2000, 2000, 2000] * 3, parents=SB)
#     MPara = Sensor.ParaM_list[:-1]  # 最后一项的时间戳就不留了
#     XPara = Sensor.ParaX_list[:-1]
#     # 该传感器模版的测量方法和测量噪声均为雷达常见的定义，无需修改
#     # 该传感器的默认参数是定义了一个EKF滤波器，使用IMM需要将其替换
#     Filter = IMMFilterModel(XPara, MPara, SB)
#     Filter.clearSubFilter()
#     for MM in [CVModel,CAModel,CTModel]:
#         Temp_Filter = BasicEKFModel(XPara,MPara,SB).loadMovementModel(MM)
#         Filter.addSubFilter(Temp_Filter)
#     Filter.DataInit(X0, np.diag([100, 4, 1]*3))
#     Filter.loadMeasureModel(Sensor.getMeasureModel())
#     Filter.loadMeasureNoiseModel(Sensor.getMeasureNoiseModel())
#
#     Sensor.setFilterModel(Filter)
#
#     true_data = TFK1.get_real_data_all()
#
#     # 参数产生
#     LambdaLib = np.arange(2, 21) * 5e-8
#     BLib = np.arange(-5, 6) * 2e11
#
#     # region [+]迭代传感器的两种方法
#     for t in range(1, Simulate_time):
#         X = np.array(true_data.iloc[t, :-1])
#         # 随机生成的参数
#         lmd = np.random.choice(LambdaLib)
#         b = np.random.choice(BLib)
#         # Xkf[:, t] = Sensor.step(X)
#         Sensor.step(X, lmd, b)
#
#     recordsA = Sensor.getRecorderData()
#     ProbInfo = Filter.getIMMProbRecorder().get_data_all()
#
#     # endregion

if __name__ == '__main__':

    from FuzzyModel.FLS import FormalNorm_layer
    from FuzzyModel.MyModel import AdoptTimeFLSLayer
    import torch
    from torch.utils.data import DataLoader
    import torch.optim.lr_scheduler as lr_scheduler
    from utils.logger import rootlogger
    from FuzzyModel.Trainer import MSETrainer
    from utils.Track_Generate import Random_Track_Generate
    from config import device as DEVICE
    device = DEVICE
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
    model = AdoptTimeFLSLayer(9, time_dim, 64, 9, 1).to(device=device)
    epoch_num = 10
    learning_rate = 10e-1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    rootlogger('Train')
    Train = MSETrainer(model=model, loader_train=train_loader, loader_test=test_loader, optimizer=optimizer,
                       lrScheduler=scheduler)

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
    data_draw4 = np.array(Fuzzy_Est_tensor[:, [0, 3, 6]].detach())
    # data_draw2 = Xkf[[0, 3, 6], :].T
    # data_draw2 = recordsA["EstimateRecorder"].iloc[:, [0, 3, 6]].to_numpy()
    # data_draw3 = recordsB["EstimateRecorder"].iloc[:, [0, 3, 6]]

    ax = plt.axes(projection='3d')


    def draw_3D(Ax, data_draw, label):
        x = data_draw[:, 0]
        y = data_draw[:, 1]
        z = data_draw[:, 2]
        Ax.plot3D(x, y, z, label=label)


    # 三维线的数据
    # draw_3D(ax,data_draw1,"real")
    # draw_3D(ax,data_draw2,"Est")
    draw_3D(ax, data_draw3, "real2")
    draw_3D(ax, data_draw4, "FuzzyEst")

    plt.legend()
    plt.show()
    # endregion
