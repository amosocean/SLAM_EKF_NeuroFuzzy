#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Message.py
# @Time      :2023/1/6 5:18 PM
# @Author    :Kinddle
"""
Message类:
用于在控制台输出信息
"""

class Message:

    @staticmethod
    def print(*args,**kwargs):
        return print(*args,**kwargs)
    @staticmethod
    def warning(*args,**kwargs):
        return print(*args,**kwargs)


