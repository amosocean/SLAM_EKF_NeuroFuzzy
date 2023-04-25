#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :gen_chaotic_time_series_data.py
# @Time      :2023/4/24 3:47 PM
# @Author    :Oliver

'''
Mackey-Glass chaotic time series
'''
import numpy as np
import math
def gen_series(tao:int, init=1, data_len=1000):
    rtn = np.zeros(data_len + 1)
    rtn[tao] = init
    for t in range(tao, data_len):
        rtn[t+1] = 0.9*rtn[t] + 0.2*rtn[t-tao]/(1+pow(rtn[t-tao],10))

    return rtn

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    tao1 = 12
    tao2 = 38
    data_len=1500
    data_from = 1000
    data_to = 1500+1
    x = np.arange(data_to-data_from)+data_from
    data1 = gen_series(tao1,data_len=data_len)
    data2 = gen_series(tao2,data_len=data_len)
    plt.subplot(2,2,1)
    # plt.plot(data1[tao1:])
    plt.plot(x,data1[data_from:data_to])
    plt.subplot(2,2,2)
    # plt.plot(data1[tao2:])
    plt.plot(x,data2[data_from:data_to])
    plt.subplot(2,2,3)
    plt.plot(data1[data_from:data_to],data1[data_from-tao1:data_to-tao1])
    # plt.plot(data1[2*tao1:],data1[tao1:-tao1])
    plt.subplot(2,2,4)
    # plt.plot(data2[2*tao2:],data2[tao2:-tao2])
    plt.plot(data2[data_from:data_to],data2[data_from-tao2:data_to-tao2])
    plt.show()




