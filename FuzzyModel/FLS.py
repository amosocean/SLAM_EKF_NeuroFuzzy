#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FLS.py
# @Time      :2023/4/25 3:34 PM
# @Author    :Oliver

from FuzzyModel.FLSMF import GaussianFunction, TrapFunction
import torch

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
                    self.sample_func = lambda x: torch.cat([x, x + torch.randn(sample_num)[:,None] * sigma], dim=-2)
                else:
                    raise RuntimeError("Use unbounded only when generating samples from normal distributions ")
            else:
                bound = abs(sample_bound)
                if sample_type == "normal":
                    sigma = bound / 3
                    self.sample_func = lambda x: torch.cat([x, x + torch.randn(sample_num)[:,None] * sigma], dim=-2)
                elif sample_type == "uniform":
                    self.sample_func = lambda x: torch.cat([x, (torch.rand(sample_num)[:,None] - 0.5) * 2 * bound], dim=-2)
                elif sample_type == "linspace":
                    self.sample_func = lambda x: torch.cat([x, (torch.linspace(-bound, bound, sample_num)[:,None])], dim=-2)

        if Fuzzifier_Function is None:
            self.Fuzzifier_Function = GaussianFunction([xDim], torch.zeros(xDim), torch.ones(xDim),FixedMean=True, FixedSigma=False)

    def forward(self, input):
        input = input.unsqueeze(-2)
        extend_x = self.sample_func(input)
        extend_Mu = self.Fuzzifier_Function(extend_x - input)
        return torch.stack([extend_x, extend_Mu]).unsqueeze(-1)

class GaussianInferenceLayer(torch.nn.Module):
    def __init__(self, xDim, rule_num, gauss_mean=None, gauss_sigma=None, mask=None):
        super().__init__()
        self.rule_num = rule_num
        self.Ant_Function = GaussianFunction([xDim,rule_num])


    def forward(self, input):
        extend_x, extend_Mu_A = input
        extend_Mu_B = self.Ant_Function(extend_x)
        raw_Mu_Q = extend_Mu_A * extend_Mu_B
        return torch.stack([extend_x * torch.ones(self.rule_num),raw_Mu_Q])
        # Mu_Q, x_idx = torch.max(extend_Mu_A * extend_Mu_B, dim=-3, keepdim=True)
        # # gather_x = torch.gather(extend_x * self.extend_one,-3,x_idx)
        # gather_x = torch.gather(torch.repeat_interleave(extend_x,self.rule_num,-1),-3,x_idx)
        # return torch.stack([Mu_Q.squeeze(-3),gather_x.squeeze(-3)])

class TrapInferenceLayer(torch.nn.Module):
    def __init__(self,xDim, rule_num, abcd=None):
        super().__init__()
        self.rule_num = rule_num
        self.Ant_Function = TrapFunction([xDim,rule_num],abcd)

    def forward(self,input):
        extend_x, extend_Mu_A = input
        extend_Mu_B = self.Ant_Function(extend_x)
        raw_Mu_Q = extend_Mu_A * extend_Mu_B
        return torch.stack([extend_x * torch.ones(self.rule_num),raw_Mu_Q])

        # Mu_Q, x_idx = torch.max(raw_Mu_Q, dim=-3, keepdim=True)
        # gather_x = torch.gather(torch.repeat_interleave(extend_x, self.rule_num, -1), -3, x_idx)
        # return torch.stack([Mu_Q.squeeze(-3), gather_x.squeeze(-3)])

class HeightDefuzzifierLayer(torch.nn.Module):
    def __init__(self,rule_num, yDim=1,height=None):
        super().__init__()
        self.rule_num = rule_num
        if height is None:
            self.para_height = torch.nn.Parameter(torch.rand([yDim,1, rule_num]))
        else:
            self.para_height = torch.nn.Parameter(height)

    def forward(self,input):
        extend_x,Mu_Q = input
        Mu_Q, x_idx = torch.max(Mu_Q, dim=-3, keepdim=True)
        gather_x = torch.gather(extend_x, -3, x_idx)

        Norm_Mu_Q = torch.prod(Mu_Q, dim=-2)
        return torch.sum(self.para_height * Norm_Mu_Q, dim=-1)/torch.sum(Norm_Mu_Q, dim=-1)


class TSDefuzzifierLayer(torch.nn.Module):
    def __init__(self,xDim, rule_num, yDim=1, C=None):
        super().__init__()
        if C is None:
            self.para_C = torch.nn.Parameter(torch.rand([yDim, xDim + 1, rule_num]))
        else:
            self.para_C = torch.nn.Parameter(C)

    def forward(self, input):
        extend_x,Mu_Q = input
        Mu_Q, x_idx = torch.max(Mu_Q, dim=-3, keepdim=True)
        gather_x = torch.gather(extend_x, -3, x_idx)

        C0 = self.para_C[:, 0]
        C_ = self.para_C[:, 1:]
        Norm_Mu_Q = torch.prod(Mu_Q, dim=-2)
        y = C0 + torch.sum(C_*gather_x,dim=-2)
        return torch.sum(y * Norm_Mu_Q, dim=-1) / torch.sum(Norm_Mu_Q, dim=-1)


if __name__ == '__main__':
    # from FLSMF import GaussianFunction, TrapFunction
    x=torch.tensor([[1,2,3,4],[4,3,2,1]])/5
    FF = FuzzifierLayer(4,10)
    GI = GaussianInferenceLayer(4,16)
    TSD = TSDefuzzifierLayer(4,16)
    y = FF(x)
    yy = GI(y)
    yyy = TSD(yy)




