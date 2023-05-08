#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Decorator.py
# @Time      :2023/4/25 9:07 PM
# @Author    :Oliver
from FuzzyModel.FLS import FixNorm_layer
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
            input = (input + self.x_offset) * self.x_scale
            rtn = raw_forward(self, input)
            return rtn / self.y_scale - self.y_offset

        cls.forward = pack_forward
        cls.set_xy_offset_scale = set_xy_offset_scale
        return cls

    return wrapper

Norm_Layer = FixNorm_layer
class Layer_Normalize_pack(object):
    def __init__(self,cls):
        self.cls = cls

        # self.Func = Func
        # self.Norm = Norm_Layer(Shape)
        pass
    def __call__(self, cls):
        pass


if __name__ == '__main__':
    import random
    # class animal:
    #     def __init__(self, func):
    #         self.func = func
    #         self.number = random.random()
    #
    #     # @wraps
    #     def __call__(self, *args, **kwargs):
    #         print(f'working here, number = {self.number}')
    #         res = self.func(*args, **kwargs)
    #         return res
    #
    #
    # @animal
    # def test(name, kind):
    #     word = f'{name} belongs to {kind}'
    #     return word
    #
    # @animal
    # def test2(name, kind):
    #     word = f'{name} belongs to {kind}'
    #     return word
    #
    #
    # A = test('cowA', 'mammals')
    # B = test('cowB', 'mammals')
    # C = test('cowC', 'mammals')
    # D = test2('cowD', 'mammals')
    # E = test2('cowE', 'mammals')
    # F = test2('cowF', 'mammals')
    # print(type(test))
    # print(A)
    # print(D)

    class animal_cls:
        def __init__(self, cls):
            # cls实例化时调用
            self.cls = cls
            self.id = random.random()
            self.raw_forward = self.cls.forward
            self.cls.forward = self.wrap_forward
        # @wraps
        def __call__(self,Norm_shape, *args, **kwargs):
            # cls初始化时调用，最好是返回初始化后的cls
            print(f'working here, number = {self.id}')
            rtn = self.cls(*args,**kwargs)
            rtn.id = random.random()
            self.norm = Norm_shape
            return rtn

        def wrap_forward(self, *args, **kwargs):
            return self.raw_forward(self.cls,*args, **kwargs)


    @animal_cls
    class IDK:
        def __init__(self,shape="default"):
            self.id = 0
            print("init...")
            print(f"shape:{shape}")
        def forward(self, x):
            print(f'working there, x={x},id={self.id}')
            return

    idk = IDK(Norm_shape="Norm_shape",shape="input_shape")
    idk2 = IDK(Norm_shape="Norm_shape2",shape="input_shape2")
    idk.forward("hi")
    idk2.forward("Hi")
    type(IDK)
