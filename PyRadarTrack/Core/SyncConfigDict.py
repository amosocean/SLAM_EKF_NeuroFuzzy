#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :SyncConfigDict.py
# @Time      :2023/1/8 2:24 AM
# @Author    :Kinddle
"""
同步配置辞典：
将字典转化成实例对象的属性
"""
class SyncConfigDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    para_list = []

    def __init__(self, init:dict=None):
        super(SyncConfigDict, self).__init__()
        if init is None:
            init = {}
        Lack = set(self.para_list)-set(init.keys())
        if Lack:
            raise IndexError(f"参数表参数不足,缺乏：{Lack}")
        if init is not None:
            self.setConfig(init)
            # self.update(init)
            # self.sync(self)

    def sync(self, other):
        for k, v in self.items():
            if isinstance(v, (list, tuple)):
                setattr(other, k, [SyncConfigDict(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(other, k, SyncConfigDict(v) if isinstance(v, dict) else v)

    def setConfig(self,Config:dict):
        """
        只是添加，原有的信息不会删除
        """
        self.update(Config)
        self.sync(self)
        return self

    def getConfig(self):
        return dict(self)


class SystemCfg(SyncConfigDict):
    para_list = ['C', 'Fs', 'Ts', 'StateDim', 'MeasurementDim', 'SimulationTimeTicks']

    def __init__(self, init=None):
        if init is None:    # 允许初始化时不定义参数
            init = dict(zip(self.para_list, [None] * len(self.para_list)))
        super(SystemCfg, self).__init__(init)


class MeasurementNoiseCfg(SyncConfigDict):
    para_list = ['R_measure', 'lmd', 'b']  # 必填项

    def __init__(self, init=None):
        if init is None:    # 允许初始化时不定义参数
            init = dict(zip(self.para_list, [None] * len(self.para_list)))
        super(MeasurementNoiseCfg, self).__init__(init)




