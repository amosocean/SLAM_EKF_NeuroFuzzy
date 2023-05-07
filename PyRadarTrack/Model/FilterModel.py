#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FilterModel.py
# @Time      :2023/1/8 11:49 PM
# @Author    :Kinddle

"""
os:实际上我还没有想好滤波器是什么意义...但是我想尽可能的拆分一下各个模块所以就闭着眼写了
滤波器：
1. 平滑测量点                    #-主要意义
2. 预测下一个点                   #-一般需要顺便实现的功能
3. 一个滤波器一般只对一条轨迹负责    # 专用性 -- 一条轨迹对应了一个滤波器
"""
import numpy as np

from ..Core import *
from .MovementModel import MovementModelFactory

FilterModelRegister = Register()
_XPara = ['px', 'vx', 'ax', 'py', 'vy', 'ay', 'pz', 'vz', 'az']
_MPara = ['rRadial', 'vRadial', 'Beta', 'Zeta']


# _timestamp = ['timestamp']

class BasicFilterModel(BasicObjectWithCfg):
    # ['px', 'vx', 'ax', 'py', 'vy', 'ay', 'pz', 'vz', 'az', 'timestamp']
    def __init__(self, XPara_list, MPara_list, parents=None):
        super(BasicFilterModel, self).__init__(parents)
        # self.PredictXRecorder = Recorder(XPara_list)
        self.FilteredXRecorder = Recorder(XPara_list)
        self.XPara_list = XPara_list
        self.MPara_list = MPara_list

        # region [+]定义一系列常用的变量名 但并非代表所有变量都会被用到
        self.X0 = self.P0 = None
        self.X = self.P = None
        self.F = self.H = None
        self.Q = self.R = None

        self.H_Func = self.R_Func = None
        self.Measure_Func = None
        # endregion

    def loadMovementModel(self, MovementModel):
        self.setF(MovementModel.F)
        self.setQ(MovementModel.Q)
        # self.F = MovementModel.F
        # self.Q = MovementModel.Q
        return self

    def loadMeasureModel(self, MeasureModel):
        self.Measure_Func = MeasureModel.MultMeasure
        self.H_Func = MeasureModel.MeasureJacobi
        return self

    def loadMeasureNoiseModel(self, MeasureNoiseModel):
        # 虽然写了这个方法 但是由于MeasureNoise有时候需要的参数Filter中基本都没有，不强迫调用
        self.R_Func = MeasureNoiseModel.getRWithPara
        return self

    # region ----[+]set函数等
    def setF(self, F):
        self.F = F
        return self

    def setQ(self, Q):
        self.Q = Q
        return self

    def setR(self, R):
        self.R = R
        return self
    # endregion

    def DataInit(self, X0, P0):
        self.X0 = self.X = X0
        self.P0 = self.P = P0
        return self

    def reset(self):
        self.X = self.X0
        self.P = self.P0

@FilterModelRegister.register("B_KF")
class BasicKFModel(BasicFilterModel):

    def __init__(self, XPara_list, MPara_list, parents=None):
        super(BasicKFModel, self).__init__(XPara_list, MPara_list, parents)
        # self.X = None
        # self.P = None
        # self.F = None
        # self.H = None
        # self.Q = None
        # self.R = None
        #
        # self.X0 = None
        # self.P0 = None

    def ModelInit(self, F, H, Q, R):
        self.F = F
        self.H = H

        self.Q = Q
        self.R = R
        return self

    def Predict(self):
        PreX = self.F.dot(self.X)
        PreP = self.F.dot(self.P).dot(self.F.T) + self.Q
        return PreX, PreP

    def Update(self, Z, *args, **kwargs):
        """
        :param Z: 测量量
        :param args: 和kwargs一道构成生成噪声矩阵R的参数
        :param kwargs:
        :return:
        """
        PreX, PreP = self.Predict()
        S = self.H.dot(PreP).dot(self.H.T) + self.R
        K = np.dot(PreP.dot(self.H.T), np.linalg.inv(S))
        Z_res = Z-self.H.dot(PreX)

        Xkf = PreX + K.dot(Z_res)
        Pkf = (np.eye(len(self.P))-K.dot(self.H)).dot(PreP)
        self.X = Xkf
        self.P = Pkf
        self.FilteredXRecorder.step(Xkf)
        return Xkf

    def step(self, Z, *args,**kwargs):
        return self.Update(Z,*args,**kwargs)

    def reset(self):
        self.DataInit(self.X0, self.P0)


@FilterModelRegister.register("B_EKF")
class BasicEKFModel(BasicKFModel):
    # EKF需要制定一个通过X产生H的雅可比矩阵函数才能使用
    def __init__(self, XPara_list, MPara_list, parents=None):
        super(BasicEKFModel, self).__init__(XPara_list, MPara_list, parents)
        # self.H_fuc = None
        # self.Measure_fun = None

    def setJacobiFun(self, Fuc):
        self.H_Func = Fuc
        return self

    def setMeasureFun(self, Fuc):
        self.Measure_Func = Fuc
        return self

    def ModelInit(self, F, H_Fuc, Q, R):
        self.F = F
        self.Q = Q
        # self.Measure_fun(Measure_Fuc)
        self.setJacobiFun(H_Fuc)
        self.R = R
        return self

    def Update(self, Z, returnDetail=False, *args, **kwargs):
        PreX, PreP = self.Predict()
        PreZ = self.Measure_Func(PreX).T[0]
        self.H = H = self.H_Func(PreX)

        S = H.dot(PreP).dot(H.T) + self.R
        K = np.dot(PreP.dot(H.T), np.linalg.inv(S))
        Z_res = Z-PreZ

        Xkf = PreX + K.dot(Z_res)
        Pkf = (np.eye(len(self.P))-K.dot(H)).dot(PreP)
        self.X = Xkf
        self.P = Pkf
        self.FilteredXRecorder.step(Xkf)
        if returnDetail:
            return Xkf, [Z_res,S]
        else:
            return Xkf

    def _ZRes(self, Z):
        PreX, PreP = self.Predict()
        PreZ = self.Measure_Func(PreX).T[0]
        # self.H = H = self.H_fuc(PreX)
        # S = H.dot(PreP).dot(H.T) + self.R
        # K = np.dot(PreP.dot(H.T), np.linalg.inv(S))
        return Z-PreZ


# @FilterModelRegister.register("FM_MM")
# class FilterModelFromMovementModel(BasicEKFModel):
#     def __init__(self, MovementModel: BaseMovementModel, *args):
#         super(FilterModelFromMovementModel, self).__init__(*args)
#         self.X = None
#         self.P = None
#         self.MovementModel = MovementModel
#
#     def init(self, X0, P0):
#         self.X = X0
#         self.P = P0
#
#     def _predict(self):
#         self.PreX = self.MovementModel.F.dot(self.X)
#         self.PreP = self.MovementModel.F.dot(self.P).dot(self.MovementModel.F.T) + self.MovementModel.Q


@FilterModelRegister.register("IMM_EKF")
class IMMFilterModel(BasicFilterModel):
    MovementModel_list = ["CVModel", "CAModel", "CTxyModel"]

    def __init__(self, XPara_list, MPara_list, parents=None):
        super(IMMFilterModel, self).__init__(XPara_list, MPara_list, parents)
        self.FilterModelNum = len(self.MovementModel_list)  # 本质上是由若干EKF子滤波器形成的
        TransferProb = 0.02
        self.IMMModelTransferMatrix = np.ones([self.FilterModelNum, self.FilterModelNum]) \
                                      * (1-TransferProb * self.FilterModelNum) \
                                      + np.eye(self.FilterModelNum)*TransferProb
        # self.IMMModelTransferMatrix = np.array([[0.96, 0.02, 0.02],
        #                                         [0.02, 0.96, 0.02],
        #                                         [0.02, 0.02, 0.96], ])
        self.IMMLikelyHoodProRecorder = Recorder(self.MovementModel_list)  # 记录混合概率
        self.IMMModelProb = np.ones(self.FilterModelNum)/self.FilterModelNum
        # self.IMMLikelyHoodPro = np.zero([3, 1])  # 记录似然函数
        self.R_Fun = None
        self.subFilters = []
        self._initSubFilters()

    def _initSubFilters(self):
        self.subFilters = []
        MovementFactory = MovementModelFactory()
        for MovementModel in self.MovementModel_list:
            Tmp_Filter = BasicEKFModel(self.XPara_list, self.MPara_list, self)
            Tmp = MovementFactory.create(MovementModel)(self.Ts, self.QSigma, -0.35)      # 写死了
            Tmp_Filter.loadMovementModel(Tmp)
            self.subFilters.append(Tmp_Filter)
        return self

    def getIMMProbRecorder(self):
        return self.IMMLikelyHoodProRecorder

    def loadMeasureModel(self, MeasureModel):
        for Filter in self.subFilters:
            Filter.Measure_Func = MeasureModel.MultMeasure
            Filter.H_Func = MeasureModel.MeasureJacobi
        return self

    def clearSubFilter(self):
        self.subFilters.clear()

    def addSubFilter(self, Filter):
        self.subFilters.append(Filter)

    def DataInit(self, X0, P0):
        for Filter in self.subFilters:
            Filter.DataInit(X0, P0)
        # self.X0 = self.X = X0
        # self.P0 = self.P = P0
        return self

    def setR(self, R):
        for Filter in self.subFilters:
            Filter.setR(R)
    # def init(self):
    #     self.X = 0

    def step(self, Z, *args,**kwargs):
        return self.Update(Z)

    def Update(self, Z):
        modelProb = self.IMMModelProb[:, None]
        c0 = modelProb * self.IMMModelTransferMatrix
        cbar = np.sum(c0, axis=0)
        mixProb = c0/cbar

        # 获取之前的信息
        tmp_X = []
        tmp_Cov = []
        for Filter in self.subFilters:
            tmp_X.append(Filter.X)
            tmp_Cov.append(Filter.P)
        tmp_X = np.array(tmp_X)
        tmp_Cov = np.array(tmp_Cov)

        tmp_Xkf = []
        tmp_Detail = []
        tmp_Prob = []
        for Filter in self.subFilters:
            # 生成针对每个模型融合的状态估计和协方差矩阵 并更新到模型中
            Xj = np.sum(tmp_X * mixProb[:,0,None],axis=0)
            res = tmp_X - Xj
            res_mat = np.array([res[i, :, None].dot(res[None, i, :]) for i in range(res.shape[0])])
            Pj = np.sum(mixProb[:, 0][:, None, None] * (tmp_Cov + res_mat), axis=0)
            Filter.X = Xj
            Filter.P = Pj
            Xkf, Detail = Filter.Update(Z, True)
            res, S = Detail
            tmp_Xkf.append(Xkf)
            d2 = res[None, :].dot(np.linalg.inv(S)).dot(res[:, None])
            # tmp_Detail.append(Detail)
            tmp_Prob.append(max(np.exp(-d2[0,0]/2)/np.sqrt(np.linalg.det(2*np.pi*S)),np.finfo(np.float64).eps))
        tmp_Prob = np.array(tmp_Prob)
        self.IMMModelProb[:] = cbar * tmp_Prob / np.sum(cbar * tmp_Prob)

        # 确认当前的信息
        tmp_X = []
        tmp_Cov = []
        for Filter in self.subFilters:
            tmp_X.append(Filter.X)
            tmp_Cov.append(Filter.P)
        tmp_X = np.array(tmp_X)
        tmp_Cov = np.array(tmp_Cov)

        Xj = np.sum(self.IMMModelProb[:, None] * tmp_X, axis=0)
        res = tmp_X - Xj
        res_mat = np.array([res[i, :, None].dot(res[None, i, :]) for i in range(res.shape[0])])
        Pj = np.sum(self.IMMModelProb[:, None, None] * (tmp_Cov + res_mat), axis=0)
        newEntropy = np.trace(Pj)
        self.FilteredXRecorder.step(Xj)
        self.IMMLikelyHoodProRecorder.step(self.IMMModelProb)
        return Xj


class FilterModelFactory(BasicFactory):
    def __init__(self):
        super(FilterModelFactory, self).__init__()
        self.service_dict = FilterModelRegister




