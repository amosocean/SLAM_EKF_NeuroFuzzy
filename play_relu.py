#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :play_relu.py
# @Time      :2023/4/25 10:37 AM
# @Author    :Oliver
import torch
from matplotlib import pyplot as plt
x = torch.linspace(-3,3,400)
F_C = torch.nn.Hardtanh(0,1)
sigma_2 = 1
F_C2 = lambda x:(torch.nn.Tanh()((x*sigma_2*2-sigma_2))+1)/2    # Tanh-> x~[-3,3] ~> y~[-1,1]
sigma_3 = 2.1
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

Approximate = False
if Approximate:
    h1= half(0,1,F_C)
    h2= half(0,1,F_C2)
    h3= half(0,1,F_C3)
    # h2 = half(0.6,0.8,reverse=True)
    # H = lambda x:h1(x)*h2(x)
    plt.plot(x,h1(x))
    plt.plot(x,h2(x))
    plt.plot(x,h3(x))
    plt.legend(["Ideal","Tanh","Sigmoid"])
    # Approximate
    plt.show()

shows = True
if shows:
    x = torch.linspace(-2, 5, 400)
    a,b,c,d = 0,2,1,3
    h1= half(a,b,F_C2)
    h2= half(d,c,F_C2)
    h3= half(a,b,F_C)
    h4= half(d,c,F_C)
    # h2 = half(0.6,0.8,reverse=True)
    # H = lambda x:h1(x)*h2(x)
    plt.plot(x,h1(x)*h2(x))
    plt.plot(x,h3(x)*h4(x))
    plt.ylim(0,1)
    # plt.plot(x,h2(x))
    # plt.plot(x,h3(x))
    plt.title(f"TrapMF({a},{b},{c},{d})")
    plt.legend(["Approximate","ideal"])
    # Approximate
    plt.show()