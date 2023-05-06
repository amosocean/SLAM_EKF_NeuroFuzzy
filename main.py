#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :train.py
# @Time      :2023/5/5 12:12 PM
# @Author    :Oliver

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from utils.dataset import MyDataset
from FuzzyModel.MyModel import FLSLayer,TSFLSLayer,TrapFLSLayer
from FuzzyModel.Trainer import BasicTrainer,MSETrainer,RMSETrainer
from utils.FuzzyPlotSupport import draw_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 100
input_dim = 2
batch_size = 5
learning_rate = 10e-5
rules_num = 16

sample_len = 5000
div_point = 4000

x1 = torch.rand(sample_len)*100
x2 = torch.rand(sample_len)*100
x = torch.stack([x1,x2])
y = ((x1+x2))**2
dataset = [[torch.stack([x1[i],x2[i]]),y[i]] for i in range(sample_len)]

train_dataset=dataset[:div_point]
train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=False)

test_dataset=dataset[div_point:]
test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False)


# customize your own model here:
# scale = max(max(test_dataset.series),max(train_dataset.series))
model = FLSLayer(input_dim,100).to(device)
# model = TrapFLSLayer(input_dim,16).to(device)
model.set_xy_offset_scale(x_scale=1/(x.max()-x.min()),x_offset=(-x.min()),
                          y_scale=1/(y.max()-y.min()),y_offset=(-y.min()))



# #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,50], gamma=0.5)

Train = RMSETrainer(model=model,loader_train=train_loader,loader_test=test_loader,optimizer=optimizer,lrScheduler=scheduler)

train_loss,test_loss = Train.run(epoch_num,2)

Train.show(train_loss,test_loss)
test_func = lambda *args:model(torch.tensor(args))

if True:
    import gif
    from utils.FuzzyPlotSupport import *
    Ant_F = model.Inference.Ant_Function
    Height = model.Defuzzifier.para_height.detach()
    sample = torch.linspace(0,1,100)[:,None,None]
    draw_data = Ant_F(sample).detach()
    @gif.frame
    def plot_(rule):
        ant = draw_data[:,:,rule]
        height = Height[:,:,rule]
        # real, pred = draw_data_pred_x[epoch]
        fig = plt.figure(figsize=[10,5])


        plt.subplot(1,2,1)
        plt.title(f"Rule-{rule + 1}")
        plt.plot(sample.squeeze(),ant)
        plt.xlim(-0.05,1.05)
        plt.ylim(0,1.5)
        plt.legend(["x{}_Ant.".format(i) for i in range(input_dim)],loc="upper right")

        plt.subplot(1,2,2)
        plt.vlines(height,0,1,label="Con.")
        plt.xlim(min(torch.min(Height), 0)-0.05,torch.max(Height)+0.05)
        plt.ylim(0,1.25)
        plt.legend()

    frame = []
    for k in range(rules_num):
        of = plot_(k)
        frame.append(of)
    frame[0].save("output/Fuzzy_para_x-y.gif",save_all=True, loop=True, append_images=frame[1:],
               duration=750, disposal=2)



