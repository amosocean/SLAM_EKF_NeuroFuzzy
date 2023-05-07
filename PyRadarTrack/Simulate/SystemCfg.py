#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :SystemCfg.py
# @Time      :2023/1/8 1:35 AM
# @Author    :Kinddle

# from ..Core import *
"""
不是很想支持嵌套，就不写那么麻烦的了
准确的说：
通过字典方法添加的健值对，只能在第一层成为SystemCfg的成员变量
通过初始化传入字典建立的SystemCfg则可以嵌套
此外在任何时候与自身同步也可以产生可嵌套的键值对

但是做个参数表也要嵌套也太..了，别吧
如果有很多地方用的到这种形式的话可能还是会抽象出来的。。
"""

# from ..Core import SyncConfigDict
#
#
# # from PyRadarTrack.Core.SyncConfigDict import SyncConfigDict
#
#
# class SystemCfg(SyncConfigDict):
#     para_list = ['C', 'Fs', 'Ts', 'StateDim', 'MeasurementDim', 'SimulationTimeTicks']
#
#     def __init__(self, init=None):
#         if init is None:    # 允许初始化时不定义参数
#             init = dict(zip(self.para_list, [None] * len(self.para_list)))
#         super(SystemCfg, self).__init__(init)
#
#
# if __name__ == '__main__':
#     # 若要运行请注释 from ..Core import *
#     # from PyRadarTrack.Core.SyncConfigDict import SyncConfigDict
#
#     class A:
#         pass
#
#
#     other = A()
#     old = dir(other)
#     print(dir(other))
#     d = {'a': 1, 'b': {'c': 2}, 'd': ["hi", {'foo': "bar"}]}
#     asas = {'C': 300000000.0, 'Fs': 1, 'Ts': 0.1, 'StateDim': 9, 'MeasurementDim': 3, 'SimulationTimeTicks': 500}
#     # x = SystemCfg(d)
#     y = SystemCfg(asas)
#     # x.update(d)
#
#     y.sync(other)
#     new = dir(other)
#     print(dir(other))
#     print(f"new:{set(new) - set(old)}")
