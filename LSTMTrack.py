#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :IMMEKF_Final.py
# @Time      :2023/1/16 8:23 PM
# @Author    :Kinddle

import numpy as np

if __name__ == '__main__':

    from FuzzyModel.FLS import FormalNorm_layer
    from FuzzyModel.MyModel import LSTMNet
    import torch
    from torch.utils.data import DataLoader
    import torch.optim.lr_scheduler as lr_scheduler
    from utils.logger import rootlogger
    from FuzzyModel.Trainer import MSETrainer
    from utils.Track_Generate import SNRNoise_Track_Dataset_Generate
    batch_size = 50
    time_dim =20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Simulate_time = 500
    TFK1 = SNRNoise_Track_Dataset_Generate(Simulate_time, seed=666, xWin=time_dim)
    TFK2 = SNRNoise_Track_Dataset_Generate(Simulate_time, seed=667, xWin=time_dim)
    # region 规划初始点和初始速度
    X0 = np.array([3300, 2, 1e-3, 3400, 3, 3e-3, 3500, 4, 4e-4])
    X1 = np.array([3300, -2, -1e-3, 3400, -3, -3e-3, 3500, -4, -4e-4])
    TFK1.gen_randomTrack(X0)
    TFK2.gen_randomTrack(X1)
    # endregion
    
    #### 数据集加入噪声
    TFK1=TFK1.add_noise(snr=-25)
    TFK2=TFK2.add_noise(snr=-25)
    ####


    train_loader = DataLoader(dataset=TFK1,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(dataset=TFK2,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    # A = Test(tensor_real_data[:time_dim])
    #model = AdoptTimeFLSLayer(9, time_dim, 64, 9, 1).to(device=device)
    model = LSTMNet(xDim=9, xTimeDim=time_dim,num_layers=5, hidden_size=8, yDim=9, yTimeDim=1).to(device=device)
    print(model.parameters)
    epoch_num = 30
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,50], gamma=0.5)
    rootlogger('Train_FuzzyTrack')
    Train = MSETrainer(model=model, loader_train=train_loader, loader_test=test_loader, optimizer=optimizer,
                       lrScheduler=scheduler,logName='Train_FuzzyTrack')

    train_loss, test_loss = Train.run(epoch_num, div=5, show_loss=True)

    Fuzzy_Est = []

    for b in test_loader:
        x = b[0].to(device)
        output = model(x).squeeze()
        Fuzzy_Est.append(output)

    Fuzzy_Est_tensor = torch.stack(Fuzzy_Est)

    # region [+]绘图
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    data_draw1 = np.array(test_loader.dataset.get_pure_track()[:, [0, 3, 6]].detach().cpu())
    data_draw2 = np.array(test_loader.dataset.get_noisy_track()[:, [0, 3, 6]].detach().cpu())
    #data_draw3 = TFK2.Track.get_real_data_all().iloc[:Simulate_time, [0, 3, 6]].to_numpy()
    data_draw4 = np.array(Fuzzy_Est_tensor[:, [0, 3, 6]].detach().cpu())
    

    ax = plt.axes(projection='3d')


    def draw_3D(Ax, data_draw, label):
        Ax.plot3D(data_draw[:, 0], data_draw[:, 1], data_draw[:, 2], label=label)


    # 三维线的数据
    draw_3D(ax,data_draw1,"real")
    draw_3D(ax,data_draw2,"Measure(Noisy)")
    #draw_3D(ax, data_draw3, "real2")
    draw_3D(ax, data_draw4, "LSTMEst")

    plt.legend()
    plt.show()
    # endregion
