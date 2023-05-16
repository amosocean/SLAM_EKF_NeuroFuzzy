#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ModelPacket.py
# @Time      :2023/5/16 2:32 PM
# @Author    :Oliver

"""
部分model由于设计上不能直接使用输入输出，在此处可以设计一个包装类实现与实际数据的适配关系

"""
import torch.nn


class BasicPacket(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def setupModel(self,model):
        self.model = model
        return self

    def forward(self,x):
        return self.model(x)
