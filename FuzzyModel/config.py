#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :config.py
# @Time      :2023/5/7 9:02 PM
# @Author    :Oliver
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)