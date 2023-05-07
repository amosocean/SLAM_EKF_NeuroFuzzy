#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2023/1/4 11:37 PM
# @Author    :Kinddle
"""
仿真类
"""

# 模型这么少不需要用Factory 写上只是以防万一
from .SimulationBox import SimulationBoxFactory
# from .Target import TargetFactory

from .SimulationBox import SimulationBox
from .Target import TargetFromKeyframe


