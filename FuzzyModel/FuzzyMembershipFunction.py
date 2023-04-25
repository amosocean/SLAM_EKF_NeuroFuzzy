#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyMembershipFunction.py
# @Time      :2023/4/20 6:50 PM
# @Author    :Oliver
import torch


def GaussianMF(mean, sigma):
    def wrap(input):
        # input = input.unsqueeze(-1)
        return torch.exp(-(input - mean) ** 2 / (2 * sigma ** 2))

    return wrap


def TrapMF(a, b, c, d):
    basic_foo = torch.nn.Hardtanh(0, 1)
    if a == b:
        a = b - 1e-6
    if c == d:
        d = c + 1e-6

    def wrap(input):
        m = basic_foo((input - a) / (b - a))
        n = basic_foo((d - input) / (d - c))
        return m * n

    return wrap


def TriMF(a,b,c):
    return TrapMF(a,b,b,c)
