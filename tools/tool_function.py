#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool_function.py
# @Time      :2023/4/15 1:19 PM
# @Author    :Kinddle
from tools import *
def normalize_angle(phi):
    """
    Normalize Phi to [-pi, pi]
    """
    # A = lambda x: x - ((x - 1) // 2 + 1) * 2
    # B = lambda x: x + ((- 1 - x) // 2 + 1) * 2
    return phi - ((phi - np.pi)//(2*np.pi) + 1)*2*np.pi

def normalize_all_bearings(z):
    for i in range(len(z)):
        if i%2 ==1:
            z[i] = normalize_angle(z[i])
    return z

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    x = np.linspace(-8,8,1000)
    A = lambda x: x - ((x - 1) // 2 + 1) * 2
    B = lambda x: x + ((- 1 - x) // 2 + 1) * 2

    # plt.subplot(2,1,1)
    # plt.plot(x,A(x))
    # plt.subplot(2,1,2)
    # plt.plot(x,B(x))
    plt.plot(x,normalize_angle(x))
    plt.show()

