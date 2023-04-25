#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyLogicSystem.py
# @Time      :2023/4/23 5:40 AM
# @Author    :Oliver

from .FuzzyMembershipFunction import *
import torch
Default_sigma = 0.2

class FuzzifierLayer(torch.nn.Module):
    def __init__(self, xDim, sample_num=0, sample_bound=None, sample_type='normal', Fuzzifier_Function=None, sigma=1):
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
        self.para_gauss_sigma = torch.nn.Parameter(torch.rand([xDim, 1, rule_num])*Default_sigma
                                                   if gauss_sigma is None else gauss_sigma)

        self.mask = torch.ones([xDim, rule_num]) if mask is None else mask
        self.Ant_Function = GaussianMF(0, 1)

    def forward(self, input):
        extend_x, extend_Mu_A = input.unsqueeze(-1)
        extend_Mu_B = self.Ant_Function((extend_x - self.para_gauss_mean) / self.para_gauss_sigma)
        Mu_Q, x_max = torch.max(extend_Mu_A * extend_Mu_B, dim=-2)
        return Mu_Q

class HeightDefuzzifierLayer(torch.nn.Module):
    def __init__(self, rule_num, yDim=1, height=None):
        """

        :param rule_num:
        :param ySampleDim:
        :param rule_y: []
        :param Defuzzifier_Function: ...y -> ...Mu(y)
        """
        super().__init__()
        if height is None:
            self.para_height = torch.nn.Parameter(torch.rand([yDim, rule_num]))
        else:
            self.para_height = torch.nn.Parameter(height)

    def forward(self,input):
        input = torch.prod(input, dim=-2)
        input = input * 1       # height
        return torch.sum(self.para_height*input, dim=-1)/torch.sum(input, dim=-1)


class DefuzzifierLayer(torch.nn.Module):
    def __init__(self, rule_num, yDim=1, ySampleDim=1000, offset=0, scale=1, rule_y=None, Defuzzifier_Function=None, Defuzzifier_Function_AsPara = False):
        """

        :param rule_num:
        :param ySampleDim:均匀采样的点数
        :param rule_y: []
        :param Defuzzifier_Function: ...y -> ...Mu(y)
        """
        super().__init__()
        self.offset = 0
        self.scale = 0
        if Defuzzifier_Function is None:
            self.Defuzzifier = GaussianMF(0, 1)
            self.para_mean = torch.nn.Parameter(torch.rand([yDim,1, rule_num]))
            self.para_sigma = torch.nn.Parameter(torch.rand([yDim,1, rule_num])*Default_sigma)

        self.y_Sample = torch.linspace(0,1, ySampleDim).repeat(yDim,1).unsqueeze(-1)

        #     pass
        # if rule_y is None:
        #     self.para_y = torch.nn.Parameter(torch.rand([yDim, ySampleDim, rule_num]))
        #     self.para_mu = torch.nn.Parameter(torch.ones([yDim, ySampleDim, rule_num]))
        # else:
        #     self.para_y = torch.nn.Parameter(rule_y)



    def forward(self,input):
        Mu_Q = torch.prod(input, dim=-2).unsqueeze(-2)
        yi_sample = self.Defuzzifier((self.y_Sample-self.para_mean)/self.para_sigma)*Mu_Q
        Mu_B_y,_ = torch.max(yi_sample,dim=-1)    #union of Mu_B^l
        # input = input * y  # height


        return torch.sum(self.y_Sample.squeeze(-1) * Mu_B_y, dim=-1) / torch.sum(Mu_B_y, dim=-1)




if __name__ == '__main__':
    from utils.gen_chaotic_time_series_data import gen_series
    tao1 = 12
    tao2 = 38
    data_len = 1500
    # data_from = 1000
    # data_to = 1500 + 1
    # data1 = gen_series(tao1, data_len=data_len)
    # data2 = gen_series(tao2, data_len=data_len)
    data = torch.tensor([[1,2,3,4],[3,4,5,6]])/10
    FF = FuzzifierLayer(4, 10)
    IL = GaussianInferenceLayer(4,16)
    DF = DefuzzifierLayer(16)

    data2 = FF(data)
    data3 = IL(data2)
    data4 = DF(data3)

    print(data4)






