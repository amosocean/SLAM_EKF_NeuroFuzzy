#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Trash_NeuroFuzzySystem.py
# @Time      :2023/4/19 3:24 PM
# @Author    :Oliver

import torch
import torch.nn.functional as F
from FuzzyModel import alphabet


class AntecedentLayer(torch.nn.Module):
    def __init__(self, xDim=3, AnntDim=3, gauss_mean=None, gauss_sigma=None):
        super().__init__()
        self.input_dim = xDim
        self.Annt_dim = AnntDim
        if gauss_mean is None:
            gauss_mean = torch.rand([self.input_dim, self.Annt_dim])
            self.para_gauss_mean = torch.nn.Parameter(gauss_mean)
        else:
            assert isinstance(gauss_mean, torch.Tensor), "gauss_mean 必须是一个张量"
            self.para_gauss_mean = torch.nn.Parameter(gauss_mean)
        if gauss_sigma is None:
            gauss_sigma = torch.rand([self.input_dim, self.Annt_dim])
            self.para_gauss_sigma = torch.nn.Parameter(gauss_sigma)
        else:
            assert isinstance(gauss_sigma, torch.Tensor), "gauss_sigma 必须是一个张量"
            self.para_gauss_sigma = torch.nn.Parameter(gauss_sigma)

    def forward(self, input):
        input = input.unsqueeze(0)
        input = torch.einsum("'...AB->AB'", input)
        input = input.unsqueeze(-1)
        input = (input - self.para_gauss_mean) / (self.para_gauss_sigma)
        input = torch.square(input) * (-0.5)
        input = torch.exp(input)
        return input


class CompleteRuleLayer(torch.nn.Module):
    def __init__(self, xDim=3, AnntDim=3, rule_tensor=None):
        super().__init__()
        # self.xDim = xDim
        # self.AnntDim = AnntDim
        # self.div_chr = ",".join(alphabet[:xDim]) + "->" + alphabet[:xDim]
        self.div_chr = "..." + ",...".join(alphabet[:xDim]) + "->" + "..." + alphabet[:xDim]
        if rule_tensor is None:
            Consequence_Hypercube = torch.rand([AnntDim] * xDim) * 10
            self.para_cons = torch.nn.Parameter(Consequence_Hypercube)
        else:
            assert isinstance(rule_tensor, torch.Tensor), "rule_tensor 必须是一个张量"
            self.para_cons = torch.nn.Parameter(rule_tensor)

    def forward(self, input):
        """
        :param input: [xDim,AnntDim,...]
        :return: [AnntDim,AnntDim,...] -> tensor with "XDim" Dimensionality totally
        """
        print(input)
        Mu = torch.einsum(self.div_chr, *input.split(1, dim=-2)).squeeze()
        return torch.sum(Mu * self.para_cons) / torch.sum(Mu)


# class CompleteRuleLayer2(torch.nn.Module):
#     def __init__(self, xDim=3, AnntDim=3,rule_tensor=None,exist_tensor=None):
#         super().__init__()
#         self.xDim = xDim
#         self.AnntDim = AnntDim
#         self.input_shape = [xDim,AnntDim]
#         self.view_base = torch.ones(xDim,xDim)-torch.eye(xDim)*2
#         self.view_base = self.view_base.to(dtype=int).tolist()
#         if rule_tensor is None:
#             Consequence_Hypercube = torch.rand([AnntDim]*xDim)*10
#             self.para_cons = torch.nn.Parameter(Consequence_Hypercube)
#         else:
#             assert isinstance(rule_tensor, torch.Tensor), "rule_tensor 必须是一个张量"
#             self.para_cons = nn.Parameter(rule_tensor)
#         # if exist_tensor is None:
#         #     self.Exist_tensor = torch.ones([AnntDim]*xDim)
#         # else:
#         #     assert isinstance(exist_tensor, torch.Tensor), "exist_tensor 必须是一个张量"
#         #     self.Exist_tensor = exist_tensor
#     def forward(self,input):
#         """
#         :param input: [xDim,AnntDim,...]
#         :return: [AnntDim,AnntDim,...] -> tensor with "XDim" Dimensionality totally
#         """
#         # view_base = [1]*self.xDim
#         xs = [input[i].view(self.view_base[i]) for i in range(self.xDim)]
#         # tmp = torch.ones([self.AnntDim]*self.xDim)
#         tmp = torch.clone(self.Exist_tensor).detach()
#         for x in xs:
#             tmp *=x
#         return torch.sum(tmp * self.para_cons)/torch.sum(tmp)

class Net1(torch.nn.Module):
    '''
    Singleton Height FLS : The most Ez one
    '''

    def __init__(self, xDim=3, AnntDim=3, gauss_mean=None, gauss_sigma=None, rule_tensor=None):
        super().__init__()
        self.FFL = AntecedentLayer(xDim, AnntDim, gauss_mean, gauss_sigma)
        self.CRL = CompleteRuleLayer(xDim, AnntDim, rule_tensor)

    def forward(self, input):
        input = self.FFL(input)
        input = self.CRL(input)
        return input


if __name__ == '__main__':
    FF = AntecedentLayer(AnntDim=5)
    CR = CompleteRuleLayer(AnntDim=5).cpu()
    # print(FF.para_gauss_mean)
    # x = torch.tensor([3.,5.,7.],requires_grad=True)/10
    # 一般0维为样本轴
    '''
    sample means bench
    • 向量数据：2D数据，形状为（samples，features）。
    • 时间序列数据 或 序列数据，3D张量，形状为（samples，timesteps，features）
    • 图像：4D张量，形状为（samples，height，width，channels ）或（samples，frames，channels，height，weight）
    • 视频：5D张量，形状为（samples，frames，height， width，channels）或（samples， frame， channels，height， width）
    '''
    x = torch.tensor([[3., 5., 7.],
                      [3., 5., 7.]], requires_grad=True) / 10
    pred = FF(x)
    # print(pred)
    cons = CR(pred)
    print(cons)
    # exp = Exp()
    # x1 = torch.tensor([3., 4.], requires_grad=True)
    # x2 = exp.apply(x1)
    # y2 = exp.apply(x2)
    #
    # y2.sum().backward()
    # print(x1.grad)
