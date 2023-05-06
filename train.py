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
from utils.logger import rootlogger
from FuzzyModel.MyModel import FLSLayer, TSFLSLayer, TrapFLSLayer
from FuzzyModel.Trainer import BasicTrainer, MSETrainer, RMSETrainer
from utils.FuzzyPlotSupport import draw_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 100
input_dim = 4
batch_size = 5
learning_rate = 10e-5
rules_num = 16

train_dataset = MyDataset(tao=38, start_index=1001, end_index=1500)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False)

test_dataset = MyDataset(tao=38, start_index=1501, end_index=1995)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=False)

# customize your own model here:
scale = max(max(test_dataset.series), max(train_dataset.series))
model = FLSLayer(input_dim, 16).to(device)
model.set_xy_offset_scale(x_scale=1 / scale, y_scale=1 / scale)

# #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50], gamma=0.5)
rootlogger('Train')
Train = RMSETrainer(model=model, loader_train=train_loader, loader_test=test_loader, optimizer=optimizer,
                    lrScheduler=scheduler, logName="Train")

train_loss, test_loss = Train.run(10, 1, True)

# log.debug('debug')
# log.info('info')
# log.warning('警告')
# log.error('报错')
# log.critical('严重')
# Train.show(train_loss,test_loss)
