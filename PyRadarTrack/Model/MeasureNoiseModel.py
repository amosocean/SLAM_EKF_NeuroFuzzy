#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MeasureNoiseModel.py
# @Time      :2023/1/8 4:45 PM
# @Author    :Kinddle

"""
噪声偶尔有动态变化的情况
想开个类，顺便在其中实现从协方差矩阵到噪声的转化
目的是通过组合MeasureModel和measureNoise实现一个有噪声的传感器仿真

"""
# import numpy as np
import numpy as np
from decimal import Decimal
from ..Core import *
import decimal

# from ..Core.SyncConfigDict import MeasurementNoiseCfg
MeasureNoiseRegister = Register()


def _svd_Sqrtm(M):
    # 对实对称方阵，使用scipy.linalg.sqrtm似乎也足够了 但偶尔有奇怪的复数进入 遂用svd
    print(M)
    u, s, v = np.linalg.svd(M)
    assert np.all(np.dot(u * np.sqrt(s), (u * np.sqrt(s)).T) - M < 1e-10)
    return u * np.sqrt(s)
    # return u.dot(np.sqrt(np.diag(s)))


def _Cholesky_real(M):
    # 对实对称方阵，使用scipy.linalg.sqrtm似乎也足够了
    from scipy.linalg import cholesky
    cholesky(M)
    u, s, v = np.linalg.svd(M)
    return u.dot(np.sqrt(np.diag(s)))


class BasicMeasureNoiseModel(BasicObject):
    MySqrtM = lambda _, M: _svd_Sqrtm(M)

    # 推荐使用R的根推测
    def __init__(self, XDim, MDim, GDriveMode=False):
        super(BasicMeasureNoiseModel, self).__init__()
        self.XDim = XDim
        self.MDim = MDim
        self.GDriveMode = GDriveMode
        if GDriveMode:
            self.G = np.zeros(shape=[MDim, XDim])
            self.R = np.eye(XDim)
            self.sqrtR = np.eye(XDim)
        else:
            self.R = np.eye(MDim)
            self.sqrtR = np.eye(MDim)

    def Noise(self, *args, **kwargs):
        if self.GDriveMode:
            return self.G.dot(self.sqrtR.dot(np.random.randn(self.XDim)))
        else:
            return self.sqrtR.dot(np.random.randn(self.MDim))

    def _setG(self, G):
        if self.GDriveMode:
            self.G[:] = G
        return self

    def _setR(self, R):
        self.R[:] = R
        self.sqrtR[:] = self.MySqrtM(R)
        assert np.all((np.dot(self.sqrtR, self.sqrtR.T) - self.R) < 1e-10)
        return self

    def setSqrtR(self, SqrtR):
        self.sqrtR[:] = SqrtR
        self.R[:] = np.dot(SqrtR, SqrtR.T)
        return self


class BasicParaSelectaModel(BasicMeasureNoiseModel):
    para_list = []

    def __init__(self, XDim, MDim, GDriveMode=False):
        super(BasicParaSelectaModel, self).__init__(XDim, MDim, GDriveMode)
        para_dict = dict(zip(self.para_list, [None] * len(self.para_list)))
        self.Cfg = SyncConfigDict(para_dict)

    def setPara(self, para_dict=None, **kwargs):
        # 优先级还是传入参数更高 字典次之
        if para_dict is not None:
            self.Cfg.setConfig(para_dict)
        self.Cfg.setConfig(kwargs)
        self.Cfg.sync(self)
        return self

    def getParaProject(self):
        return self.Cfg


@MeasureNoiseRegister.register("Liner")
class LinerMeasureNoiseModel(BasicMeasureNoiseModel):
    def __init__(self, XDim, MDim, GDriveMode=False):
        super(LinerMeasureNoiseModel, self).__init__(XDim, MDim, GDriveMode)


@MeasureNoiseRegister.register("PSM_GPLFM")
class PSModelGPLFM(BasicParaSelectaModel):
    para_list = ["fc", "wc", "R0", "SigmaBeta", "SigmaZeta", "Km", ]

    def __init__(self, XDim, MDim, GDriveMode=False):
        super(PSModelGPLFM, self).__init__(XDim, MDim, GDriveMode)
        self.setPara({"fc": 4e12, "wc": 2 * np.pi * 4e12,
                      "R0": 7e3, "SigmaBeta": 3,
                      "SigmaZeta": 3, "Km": 100, })

        # self.MNCfg = SyncConfigDict(self.para_dict)  # 自己保存一个Cfg 这样不必每次都完全更新其内容

    def Noise(self, MNCfg_dict=None, **kwargs):
        Tmp_MNCfg = MeasurementNoiseCfg(dict({} if MNCfg_dict is None else MNCfg_dict, **kwargs))
        self.sqrtR[:] = self._getSqrtRWithPara(Tmp_MNCfg)
        self.R = np.dot(self.sqrtR, self.sqrtR.T)
        return super(PSModelGPLFM, self).Noise()

    # def setPara(self, para_dict):
    #     super(PSModelGPLFM, self).setPara(para_dict)

    # def _getNoiseWithPara(self, MNCfg_: [MeasurementNoiseCfg]):
    #     """
    #     :param MNCfg: 形成噪声的参数列表, 包括*R,C,*lmd,*b,[R0,wc,Km,SigmaBeta,SigmaZeta]
    #     其中方括号的信息在本类中有定义默认参数,*表示每轮测量都应当更新
    #     R表示雷达与目标之间的距离
    #     C在SystemCfg中出现过，不是很清楚是什么
    #     lmd和b就是雷达波形参数
    #     :return:Noise
    #     """
    #
    #     SqrtR = self._getSqrtRWithPara(MNCfg_)
    #     self.setSqrtR(SqrtR)
    #     return self.Noise()
    #
    # def _getRWithPara(self, MNCfg_: [MeasurementNoiseCfg]):
    #     # if self.
    #     # MNCfg_ = {k: Decimal(v) for k,v in MNCfg_.items()}
    #     MNCfg = self.MNCfg.setConfig(MNCfg_)
    #
    #     SNR = (MNCfg.R0 / MNCfg.R) ** 4
    #     R = np.zeros([self.MDim] * 2)
    #     # R2 = np.zeros([self.MDim] * 2)
    #
    #     # C =
    #
    #     R[:2, :2] = (MNCfg.C * MNCfg.lmd) ** 2 / (2 * SNR * MNCfg.wc ** 2) \
    #                 * np.array([[MNCfg.wc ** 2, -2 * MNCfg.b * MNCfg.wc],
    #                             [-2 * MNCfg.b * MNCfg.wc, (1 + 4 * MNCfg.b ** 2)]])
    #     R[2:, 2:] = np.diag([MNCfg.SigmaBeta, MNCfg.SigmaZeta]) ** 2 / (MNCfg.Km ** 2 * SNR)
    #
    #     # R2[:] = np.array(
    #     #     [[(MNCfg.C * MNCfg.lmd) ** 2 / 2 / SNR, -(MNCfg.C * MNCfg.lmd) ** 2 * MNCfg.b / MNCfg.wc / SNR, 0, 0],
    #     #      [-(MNCfg.C * MNCfg.lmd) ** 2 * MNCfg.b / MNCfg.wc / SNR,
    #     #       (MNCfg.C / MNCfg.wc) ** 2 / SNR * (2 * (MNCfg.b * MNCfg.lmd)**2 +( MNCfg.lmd ** 2))/2, 0, 0],
    #     #      [0, 0, MNCfg.SigmaBeta ** 2 / MNCfg.Km ** 2 / SNR, 0],
    #     #      [0, 0, 0, MNCfg.SigmaZeta ** 2 / MNCfg.Km ** 2 / SNR]])
    #     # assert np.all(R == R2)
    #     return R
    #
    def _getSqrtRWithPara(self, MNCfg_: MeasurementNoiseCfg):
        MNCfg = self.Cfg.setConfig(MNCfg_)

        SNR = (MNCfg.R0 / MNCfg.R_measure) ** 4
        tmp_p = (MNCfg.C * MNCfg.lmd / MNCfg.wc) / (np.sqrt(2 * SNR))
        SqrtR = np.zeros([self.MDim] * 2)

        SqrtR[:2, :2] = np.array([[MNCfg.wc, 0],
                                  [-2 * MNCfg.b, 1]]) * tmp_p
        SqrtR[2:, 2:] = np.diag([MNCfg.SigmaBeta, MNCfg.SigmaZeta]) / (MNCfg.Km * np.sqrt(SNR))
        return SqrtR

    def getRWithPara(self, MNCfg_: MeasurementNoiseCfg):
        sqrtR = self._getSqrtRWithPara(MNCfg_)
        return np.dot(sqrtR, sqrtR.T)


class MeasureNoiseModelFactory(BasicFactory):
    def __init__(self):
        super(MeasureNoiseModelFactory, self).__init__()
        self.service_dict = MeasureNoiseRegister
