#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :BasicObject.py
# @Time      :2023/1/5 2:19 PM
# @Author    :Kinddle
"""
基本对象：
包内所有的对象基本上都继承自基本对象。以防万一要实现整体层面上的功能
基本参数伴随对象：
如果有系统需要继承父类的参数，参数伴随对象可以在不特别声明的情况下令子类继承父类的参数。此处的参数特指SyncConfigDict类携带的参数。
"""
from .SyncConfigDict import SyncConfigDict


class BasicObject(object):
    def __init__(self,*args,**kwargs):
        pass

    def setAttrCfg(self, cfg_dict=None, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        if cfg_dict is not None:
            for k, v in cfg_dict.item():
                self.__setattr__(k, v)


class BasicObjectWithCfg(BasicObject):
    """
    适用于任何有系统参数需要继承的类型
    """
    # 建议声明子类来获得更好的体验
    ConfigClass = SyncConfigDict

    def __init__(self, parents, *args):
        super(BasicObjectWithCfg, self).__init__(*args)
        self.SystemCfg = self.ConfigClass()
        if parents is not None:
            if type(parents) == dict:
                self.loadSystemCfg(parents)
            else:
                # 这里比较耦合。。
                self.loadSystemCfg(parents.SystemCfg)

    def loadSystemCfg(self, SystemCfg_):
        self.SystemCfg.setConfig(SystemCfg_)
        self.SystemCfg.sync(self)