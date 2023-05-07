#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :BasicFactory.py
# @Time      :2023/1/6 5:18 PM
# @Author    :Kinddle
"""
基本工厂类：
根据设定的键值定向到不同的类对象，便于快速创建同一种类对象的不同实现
"""
from .BasicObject import BasicObject
from ..Interface import Message


class BasicFactory(BasicObject):
    service_dict = {}

    def __init__(self):
        super(BasicFactory, self).__init__()

    def create(self, key):
        """
        返回的是类的创建符
        :param key: 用以区分类的关键词
        :return: class type 可以调用以产生类
        """
        if key in self.service_dict.keys():
            return self.service_dict[key]
        else:
            Message.print(f"wrong key, we have {self.service_dict}")
            return None

    def get_keys(self):
        return self.service_dict.keys()