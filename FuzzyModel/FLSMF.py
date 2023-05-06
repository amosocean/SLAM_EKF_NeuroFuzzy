#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FLSMF.py
# @Time      :2023/4/25 3:34 PM
# @Author    :Oliver

import torch
import re

Slope_Core_HardTanh = torch.nn.Hardtanh(0, 1)
Slope_Core_Tanh = lambda x: (torch.nn.Tanh()((x * 2 - 1)) + 1) / 2
Slope_Core_Sigmoid = lambda x: (torch.nn.Sigmoid()((x * 4 - 2)))
Slope_Core_dict = {
    # "HardTanh": Slope_Core_HardTanh,
    "Tanh": Slope_Core_Tanh,
    "Sigmoid": Slope_Core_Sigmoid
}


class BasicFunction(torch.nn.Module):
    def __init__(self, input_shape, **kwargs):
        """
        kwargs =  { P_a:[...], (FixedPa:True/False),
                    P_b:[...], (FixedPb:True/False),...}
        ! set attribute
            self.para_Pa = [...] if FixedPa else torch.nn.Parameter([...])
        notice:
            when value of Pk is None, replace it with torch.rand(input_shape)
            default FixedPk is False
        """
        super().__init__()
        value_dict = {}
        Fixed_dict = {}
        for k, v in kwargs.items():
            if re.match("Fixed", k):
                Fixed_dict.update({re.match("Fixed(.*)", k).group(1): v})
            else:
                value_dict.update({k: v})
                if k not in Fixed_dict:
                    Fixed_dict.update({k: False})
        for k in value_dict.keys():
            para_name = k
            para_value = torch.rand(input_shape) if value_dict[k] is None else value_dict[k]
            para_Fixed = Fixed_dict[k]
            self.__setattr__("para_" + para_name, para_value if para_Fixed else torch.nn.Parameter(para_value))

    def forward(self, x):
        return x


class GaussianFunction(BasicFunction):

    def __init__(self, input_shape, mean=None, sigma=None, FixedSigma=False, FixedMean=False):
        super().__init__(input_shape, Mean=mean, Sigma=sigma, FixedSigma=FixedSigma, FixedMean=FixedMean)

    def forward(self, x):
        return torch.exp(-(x - self.para_Mean) ** 2 / (2 * self.para_Sigma ** 2))


class TrapFunction(BasicFunction):
    def __init__(self, input_shape, abcd=None, FixedA=False, FixedB=False, FixedC=False, FixedD=False,
                 Slope_Core="Tanh"):
        trap_abcd, _ = torch.sort(torch.rand([4, *input_shape]), dim=0) if abcd is None else (abcd, 0)
        a, b, c, d = trap_abcd
        super().__init__(input_shape, A=a, B=b, C=c, D=d,
                         FixedA=False, FixedB=False, FixedC=False, FixedD=False)
        self.Core = Slope_Core_dict[Slope_Core]

    def forward(self, x):
        m = self.Core((x - self.para_A) / (self.para_B - self.para_A))
        n = self.Core((x - self.para_D) / (self.para_C - self.para_D))
        return m * n


class HalfTrap(BasicFunction):
    def __init__(self, input_shape, ab=None, FixedA=False, FixedB=False,
                 Slope_Core="Tanh"):

        trap_abcd = torch.rand([2, *input_shape]) if ab is None else ab
        a, b = trap_abcd
        super().__init__(input_shape,A=a,B=b,FixedA=False, FixedB=False)

        self.para_A = a if FixedA else torch.nn.Parameter(a)
        self.para_B = b if FixedB else torch.nn.Parameter(b)
        self.Core = Slope_Core_dict[Slope_Core]

    def forward(self, x):
        m = self.Core((x - self.para_A) / (self.para_B - self.para_A))
        return m


class StrictlyTrapFunction(BasicFunction):
    def __init__(self, input_shape, center, slope_up, topPlat_len,  slope_down,
                 Slope_Core="Tanh"):
        super().__init__(input_shape,center=center,slope_up=slope_up,topPlat_len=topPlat_len,slope_down=slope_down)
        self.Core = Slope_Core_dict[Slope_Core]

    def forward(self, x):
        slope_up = torch.exp(self.para_slope_up)
        slope_down = -torch.exp(self.para_slope_down)
        center = self.para_center
        topPlat_len = torch.nn.LeakyReLU()(self.para_topPlat_len)
        m = self.Core(1 + slope_up*(x-center+topPlat_len/2))
        n = self.Core(1 + slope_down*(x-center-topPlat_len/2))
        return m * n
