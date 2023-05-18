#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2023/4/19 3:23 PM
# @Author    :Kinddle
from .Trainer import *
from .MyModel import *
from .config import *
# from config import device
alphabet = "".join([chr(i) for i in list(range(65, 90)) + list(range(97, 123))])
# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

