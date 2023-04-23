#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_of_optim.py
# @Time      :2023/4/23 4:06 AM
# @Author    :Oliver
"""
try to learn sth about loss and optim
"""

import torch
import numpy
from matplotlib import pyplot as plt

def FuzzifierFoo(mean,sigma):
    def wrap(input):
        input = input.unsqueeze(-1)
        return torch.exp(-(input-mean)**2/(2*sigma**2))
    return wrap

def MembershipFoo(a,b,c,d):
    basic_foo = torch.nn.Hardtanh(0,1)
    if a==b:
        a = b - 1e-6
    if c==d:
        d = c + 1e-6
    def wrap(input):
        m = basic_foo((input - a)/(b-a))
        n = basic_foo((d - input)/(d-c))
        return m*n
    return wrap


if __name__ == '__main__':
    sample = torch.linspace(-10,10,1000)

    x_in = torch.rand(4)*10 - 5    # uniform distribution ~[-5,5]
    Ff = FuzzifierFoo(x_in,0.64)
    MF = MembershipFoo(-4,-0.25,0.25,4)


    Fuzzy_in = Ff(sample)
    Mu_Fuzzy_in = Ff(sample)*MF(sample).unsqueeze(-1)

    plt.figure()
    plt.subplot(2,2,1)
    plt.vlines(x_in,0,1,colors="y",linewidth=1)
    plt.plot(sample,Ff(sample))
    plt.subplot(2,2,2)
    plt.plot(sample,MF(sample))

    plt.subplot(2,2,3)
    MFf_foo = lambda x:torch.diag(Ff(x))*MF(x)
    real_max_line = sample[torch.argmax(Mu_Fuzzy_in, dim=0)]
    plt.plot(sample,Mu_Fuzzy_in)
    plt.vlines(x_in,0,MFf_foo(x_in),colors="y",linewidth=1)
    plt.vlines(real_max_line,0,MFf_foo(real_max_line),colors="r",linewidth=1)
    plt.show()



