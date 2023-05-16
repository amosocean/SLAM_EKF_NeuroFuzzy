#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :train.py
# @Time      :2023/5/5 12:12 PM
# @Author    :Oliver

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from utils.dataset import Mackey_Glass_Dataset
from FuzzyModel.MyModel import FLSLayer,TSFLSLayer,TrapFLSLayer
from FuzzyModel.Trainer import BasicTrainer,MSETrainer,RMSETrainer
from utils.FuzzyPlotSupport import draw_loss


model = MSETrainer.LoadModelByJson("/Users/kinddlelee/PycharmProject/SLAM_EKF_NeuroFuzzy/SavedModel/2023-05-16__22-01-05/model_config.json")

