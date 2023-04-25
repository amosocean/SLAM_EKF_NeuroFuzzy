#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :play_relu.py
# @Time      :2023/4/25 10:37 AM
# @Author    :Oliver
import torch
from matplotlib import pyplot as plt
x = torch.linspace(-3,3,400)
F_C = torch.nn.Hardtanh(0,1)
sigma_2 = 3/3
F_C2 = lambda x:(torch.nn.Tanh()((x*sigma_2*2-sigma_2))+1)/2    # Tanh-> x~[-3,3] ~> y~[-1,1]
sigma_3 = 5/2.5
F_C3 = lambda x: (torch.nn.Sigmoid()((x*sigma_3*2-sigma_3)))  # Sigmoid -> x~[-5,5] ~> y~[0,1]
# adopt
# def half(left,right,reverse=False):
#     if reverse:
#         lamb = -1
#         scale = right - left
#         offset = right
#     else:
#         lamb = 1
#         scale = right-left
#         offset = -left
#     def wrap(input):
#         return F((lamb*input+offset)/scale)
#     return wrap

def half(zero,one,core):
    scale = one-zero
    offset = -zero
    def wrap(input):
        return core((input+offset)/scale)
    return wrap


h1= half(0,1,F_C)
h2= half(0,1,F_C2)
h3= half(0,1,F_C3)
# h2 = half(0.6,0.8,reverse=True)
# H = lambda x:h1(x)*h2(x)
plt.plot(x,h1(x))
plt.plot(x,h2(x))
plt.plot(x,h3(x))
plt.legend(["h1","h2","h3"])
plt.show()

