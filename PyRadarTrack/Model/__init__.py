#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2023/1/5 6:56 PM
# @Author    :Kinddle
"""
模型类：
用于管理在目标跟踪任务中可能用到的所有模型类，每个类最终都以工厂的形式实现并在此给出他们的工厂类
在具体使用某个本包包含的模型时，需要先声明工厂，然后根据对应模型的名称直接调用（可以通过get_keys函数来查看目前可用的模型）
在使用自己定义的模型时，可以将模型注册到工厂中然后在后续代码任意地方调用。
"""
from .FilterModel import FilterModelFactory
from .MeasureModel import MeasureFactory
from .MeasureNoiseModel import MeasureNoiseModelFactory
from .MovementModel import MovementModelFactory
from .SensorModel import SensorModelFactory

# Total_factory = {"MovementModelFactory": MovementModelFactory(),
#                  "MeasureNoiseModelFactory": MeasureNoiseModelFactory(),
#                  "SensorModelFactory": SensorModelFactory(),
#                  "FilterModelFactory": FilterModelFactory(),
#                  # "MovementModelFactory": MovementModelFactory(),
#                  }
