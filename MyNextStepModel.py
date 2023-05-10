#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MyNextStepModel.py
# @Time      :2023/5/10 1:29 PM
# @Author    :Oliver
from FuzzyModel.FLS import *
from FuzzyModel.Decorator import *
from config import device

from utils.Track_Generate import Random_Track_Generate

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Simulate_time = 500
TFK1 = Random_Track_Generate(Simulate_time, seed=None)
TFK2 = Random_Track_Generate(Simulate_time, seed=None)
Models = TFK1.MovementModels





