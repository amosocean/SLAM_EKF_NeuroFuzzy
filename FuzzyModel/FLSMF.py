#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FLSMF.py
# @Time      :2023/4/25 3:34 PM
# @Author    :Oliver

import torch

Slope_Core_HardTanh = torch.nn.Hardtanh(0, 1)
Slope_Core_Tanh = lambda x: (torch.nn.Tanh()((x * 2 - 1)) + 1) / 2
Slope_Core_Sigmoid = lambda x: (torch.nn.Sigmoid()((x * 4 - 2)))
Slope_Core_dict = {"HardTanh": Slope_Core_Tanh,
                   "Tanh": Slope_Core_Tanh,
                   "Sigmoid": Slope_Core_Sigmoid}


class GaussianFunction(torch.nn.Module):

    def __init__(self, input_shape, mean=None, sigma=None, FixedSigma=False, FixedMean=False):
        super().__init__()
        gauss_mean = torch.rand(input_shape) if mean is None else mean
        gauss_sigma = torch.rand(input_shape) if sigma is None else sigma

        self.para_mean = gauss_mean if FixedMean else torch.nn.Parameter(gauss_mean)
        self.para_sigma = gauss_sigma if FixedSigma else torch.nn.Parameter(gauss_sigma)

    def forward(self, input):
        return torch.exp(-(input - self.para_mean) ** 2 / (2 * self.para_sigma ** 2))


class TrapFunction(torch.nn.Module):
    def __init__(self, input_shape, abcd=None, FixedA=False, FixedB=False, FixedC=False, FixedD=False,
                 Slope_Core="HardTanh"):
        super().__init__()
        trap_abcd, _ = torch.sort(torch.rand([4, *input_shape]), dim=0) if abcd is None else (abcd,0)
        a, b, c, d = trap_abcd

        self.para_a = a if FixedA else torch.nn.Parameter(a)
        self.para_b = b if FixedB else torch.nn.Parameter(b)
        self.para_c = c if FixedC else torch.nn.Parameter(c)
        self.para_d = d if FixedD else torch.nn.Parameter(d)

        self.Core = Slope_Core_dict[Slope_Core]

    def forward(self, input):
        m = self.Core((input - self.para_a) / (self.para_b - self.para_a))
        n = self.Core((input - self.para_d) / (self.para_c - self.para_d))
        return m * n
