#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :IMMEKF_Final.py
# @Time      :2023/1/16 8:23 PM
# @Author    :Kinddle

import numpy as np

from config import getStrTime

if __name__ == '__main__':

    from FuzzyModel.FLS import FormalNorm_layer
    from FuzzyModel.MyModel import AdoptTimeFLSLayer,AdoptTimeFLSLayer_Dense,PackingAdoptTimeFLSLayer
    import torch
    from torch.utils.data import DataLoader,ConcatDataset
    import torch.optim.lr_scheduler as lr_scheduler
    from utils.logger import rootlogger,MarkdownEditor
    from FuzzyModel.Trainer import MSETrainer
    from utils.Track_Generate import SNRNoise_Track_Dataset_LinerMeasure,CovarianceNoise_Track_Dataset_LinerMeasure
    from FuzzyModel.FLS import NormalizePacking
    import torch.nn.functional as F
    batch_size = 5000
    time_dim = 50
    rule_num = 64
    snr_db = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Simulate_time = 500
    dt = 0.1
    
    Train_Dataset_List=[]
    for x in range(20):
        dataset=SNRNoise_Track_Dataset_LinerMeasure(Simulate_time, seed=x, xWin=time_dim)
        # region 规划初始点和初始速度
        X0 = np.array([3300, 2, 1e-3, 3400, 3, 3e-3, 3500, 4, 4e-4])
        dataset.gen_randomTrack(X0)
        # endregion
        #### 数据集加入噪声
        dataset=dataset.add_noise(snr=snr_db).normalize()
        ####
        Train_Dataset_List.append(dataset)
    
    Test_Dataset_List=[]
    for x in range(667,677):
        dataset=SNRNoise_Track_Dataset_LinerMeasure(Simulate_time, seed=x, xWin=time_dim)
        # region 规划初始点和初始速度
        X1 = np.array([3300, -2, -1e-3, 3400, -3, -3e-3, 3500, -4, -4e-4])
        dataset.gen_randomTrack(X1)
        # endregion
        #### 数据集加入噪声
        dataset=dataset.add_noise(snr=snr_db).normalize()
        ####
        Test_Dataset_List.append(dataset)
    
    TFK1=ConcatDataset(Train_Dataset_List)
    TFK2=ConcatDataset(Test_Dataset_List)
    # TFK1 = Random_Track_Dataset_Generate(Simulate_time,seed=666,xWin=time_dim)
    # TFK2 = Random_Track_Dataset_Generate(Simulate_time,seed=667,xWin=time_dim)

    # #### 数据集加入噪声
    # TFK1=TFK1.add_noise(snr=-25)
    # TFK2=TFK2.add_noise(snr=-25)
    # ####


    train_loader = DataLoader(dataset=TFK1,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(dataset=TFK2,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    # A = Test(tensor_real_data[:time_dim])
    #model = AdoptTimeFLSLayer(9, time_dim, 64, 9, 1).to(device=device)

    class TestModel(torch.nn.Module):
        def __init__(self,time_dim,rule_num=64):
            super(TestModel,self).__init__()
            self.NormPack = NormalizePacking(self.forward,time_dim,channel_num=9)
            self.forward = self.NormPack.forward
            self.order=time_dim
            self.x_pre=torch.zeros(self.order,9,time_dim).to(device)
            coefficient=torch.rand(self.order+1).to(device)
            self.coefficient=torch.nn.Parameter(coefficient)
            self.coefficient.requires_grad_(True)
        def forward(self,x):
            # x=torch.sum(x,dim=-1,keepdim=True)/(x.shape[-1]) #滑动平均
            # x_select=
            x=x[0]
            c=self.coefficient
            temp=x
            self.x_pre=torch.cat((self.x_pre,torch.unsqueeze(x,0)),dim=0)
            for t in range(self.order):
                x=x+c[t+1]*self.x_pre[t+1]
            rtn=x[:,-1]
            return torch.unsqueeze(rtn,-1)

    model = TestModel(time_dim=time_dim,rule_num=rule_num).to(device=device)
    print(model.parameters)
    epoch_num = 5
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,50], gamma=0.5)
    ME = MarkdownEditor().init_By_logger(rootlogger('Train_FuzzyTrack'))
    Train = MSETrainer(model=model, loader_train=train_loader, loader_test=test_loader, optimizer=optimizer,
                       lrScheduler=scheduler,logName='Train_FuzzyTrack')

    Train.run(epoch_num, div=5, show_loss=False)

    ME.add_figure("lossPic.png",figData=Train.drawLossFig(),
                  describe="### The loss of last epoch.")

    test_loader = DataLoader(dataset=Test_Dataset_List[0],
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

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


    data_draw1 = np.array(Test_Dataset_List[0].get_pure_track()[[0, 3, 6]].detach().cpu())
    data_draw2 = np.array(Test_Dataset_List[0].get_measure()[[0, 3, 6]].detach().cpu())
    #data_draw3 = TFK2.Track.get_real_data_all().iloc[:Simulate_time, [0, 3, 6]].to_numpy()
    data_draw4 = np.array(Fuzzy_Est_tensor[:,[0, 3, 6]].T.detach().cpu())
    fig = plt.figure(figsize=[16,12])
    ax = plt.axes(projection='3d')

    def draw_3D(Ax, data_draw, label):
        Ax.plot3D(*data_draw, label=label)


    # 三维线的数据
    draw_3D(ax,data_draw2,"Measure(Noisy)")
    #draw_3D(ax, data_draw3, "real2")
    draw_3D(ax, data_draw4, "FuzzyEst")
    draw_3D(ax,data_draw1,"real")

    plt.legend()
    #ME.add_figure("1.png",fig)
    plt.show()

    fig2 = plt.figure()
    data_draw_1 = data_draw1
    data_draw_3 = data_draw4
    data_draw_2 = data_draw2
    Win = time_dim
    x = torch.arange(Simulate_time)*dt
    plt.subplot(2,2,1)
    plt.plot(x, data_draw_2[0],label="Measure")
    plt.plot(x[Win-25:-25], data_draw_3[ 0],label="Est")
    plt.plot(x, data_draw_1[0], label="True")
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(x, data_draw_2[1],label="Measure")
    plt.plot(x[Win-25:-25], data_draw_3[ 1],label="Est")
    plt.plot(x, data_draw_1[1], label="True")
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(x, data_draw_2[2],label="Measure")
    plt.plot(x[Win-25:-25], data_draw_3[ 2],label="Est")
    plt.plot(x, data_draw_1[2], label="True")
    plt.legend()
    #ME.add_figure("2.png",fig)

    plt.show()


    # endregion

    # ME.saveMD()
    # if input("\n[!]是否保存模型？[y/n]") == "y":
    #     savePath = Train.ModelSave(ME.log_StartTime)
    #     print(f"Saved to {savePath} ...")
    #     ME.add_line("Model have saved to {savePath} ...")

    #     model2 = MSETrainer.LoadModelByJson(savePath)
