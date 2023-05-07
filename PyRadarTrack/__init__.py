#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2023/1/4 10:54 PM
# @Author    :Kinddle

"""
一个利用Python解决雷达跟踪问题的包，实现一个程序框架，可以装载多种技术方法实现多目标跟踪的仿真；
并在适配后可以适应雷达采集信号的离线乃至实时处理。
"""

# from Core import *
from . import Model
from . import Reality
from . import Simulate
from . import Scene
from .Core.SyncConfigDict import *






