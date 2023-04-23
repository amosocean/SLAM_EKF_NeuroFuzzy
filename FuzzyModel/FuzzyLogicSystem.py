#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyLogicSystem.py
# @Time      :2023/4/23 5:40 AM
# @Author    :Oliver

from .FuzzyMembershipFunction import *
import torch


class FuzzifierLayer(torch.nn.Module):
    def __init__(self, xDim, sample_num=0, sample_bound=None, sample_type='linspace', Fuzzifier_Function=None, sigma=1):
        """
        :param xDim:  the dimension of data input
        :param sample_num: if equal to 0, Singleton Fuzzifier, else non-Singleton; default 0
        :param sample_bound: the bound of sample, meaning the simples belong to [x-bound, x+bound]. None means unbounded
        :param sample_type:'uniform', 'normal' or 'linspace' , only normal can be used without bound assign
        :param Fuzzifier_Function: to calculate Mu(x)
        torch.cat([d,d+torch.linspace(*[-5,5],4)],dim=-1)
        """
        super().__init__()
        self.xDim = xDim
        if sample_num == 0:
            self.sample_func = lambda x: x
        else:
            if sample_bound is None:
                if sample_type == "normal":
                    self.sample_func = lambda x: torch.cat([x, x + torch.randn(sample_num) * sigma], dim=-1)
                else:
                    raise RuntimeError("Use unbounded only when generating samples from normal distributions ")
            else:
                bound = abs(sample_bound)
                if sample_type == "normal":
                    sigma = bound / 3
                    self.sample_func = lambda x: torch.cat([x, x + torch.randn(sample_num) * sigma], dim=-1)
                elif sample_type == "uniform":
                    self.sample_func = lambda x: torch.cat([x, (torch.rand(sample_num) - 0.5) * 2 * bound], dim=-1)
                elif sample_type == "linspace":
                    self.sample_func = lambda x: torch.cat([x, (torch.linspace(-bound, bound, sample_num))], dim=-1)

        if Fuzzifier_Function is None:
            self.Fuzzifier_Function = GaussianMF(0, 1)

    def forward(self, input):
        input = input.unsqueeze(-1)
        extend_x = self.sample_func(input)
        extend_Mu = self.Fuzzifier_Function(extend_x - input)
        return torch.stack([extend_x, extend_Mu])


class GaussianInferenceLayer(torch.nn.Module):
    def __init__(self, xDim, rule_num, gauss_mean=None, gauss_sigma=None, mask=None):
        super().__init__()

        self.para_gauss_mean = torch.nn.Parameter(torch.rand([xDim, 1, rule_num])
                                                  if gauss_mean is None else gauss_mean)
        self.para_gauss_sigma = torch.nn.Parameter(torch.rand([xDim, 1, rule_num])
                                                   if gauss_sigma is None else gauss_sigma)

        self.mask = torch.ones([xDim, rule_num]) if mask is None else mask
        self.Ant_Function = GaussianMF(0, 1)

    def forward(self, input):
        extend_x, extend_Mu_A = input.unsqueeze(-1)
        extend_Mu_B = self.Ant_Function((extend_x - self.para_gauss_mean) / self.para_gauss_sigma)
        Mu_Q, x_max = torch.max(extend_Mu_A * extend_Mu_B, dim=-2)
        return Mu_Q
