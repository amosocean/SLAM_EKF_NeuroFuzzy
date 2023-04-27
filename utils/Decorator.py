#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Decorator.py
# @Time      :2023/4/25 9:07 PM
# @Author    :Oliver

def scale(x_offset_=0, x_scale_=1, y_offset_=0, y_scale_=1):
    def wrapper(cls):
        cls.x_scale = x_scale_
        cls.x_offset = x_offset_
        cls.y_scale = y_scale_
        cls.y_offset = y_offset_
        raw_forward = cls.forward

        def set_xy_offset_scale(self, x_offset=None, x_scale=None, y_offset=None, y_scale=None):
            self.x_offset = x_offset if x_offset is not None else self.x_offset
            self.y_offset = y_offset if y_offset is not None else self.y_offset
            self.x_scale = x_scale if x_scale is not None else self.x_scale
            self.y_scale = y_scale if y_scale is not None else self.y_scale

        def pack_forward(self, input):
            input = input * self.x_scale + self.x_offset
            rtn = raw_forward(self, input)
            return rtn * self.y_scale + self.y_offset

        cls.forward = pack_forward
        cls.set_xy_offset_scale = set_xy_offset_scale
        return cls

    return wrapper
