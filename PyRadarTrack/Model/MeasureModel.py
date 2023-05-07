#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MeasureModel.py
# @Time      :2023/1/11 10:33 AM
# @Author    :Kinddle
"""
将测量单独领出来
目的是通过组合MeasureModel和measureNoise实现一个有噪声的传感器仿真
实现功能的尽可能分离
"""
from ..Core import *

MeasureModelRegister = Register()


class BasicMeasure(BasicObject):
    para_list = []

    def __init__(self, XDim, StationLocation=None):
        MDim = len(self.para_list)
        if StationLocation is None:
            self.StationLocation = np.zeros(XDim)
        else:
            self.StationLocation = np.array(StationLocation)
        super(BasicMeasure, self).__init__()
        self.StateDim = XDim
        self.MeasurementDim = MDim

    def MultMeasure(self, StateVectors):
        rtn = np.zeros(shape=[self.MeasurementDim, StateVectors.shape[1]])
        for t in range(len(StateVectors)):
            X = StateVectors[:, t]
            rtn[:, t] = self.SingleMeasure(X)
        return rtn

    def SingleMeasure(self, StateVector):
        return StateVector


@MeasureModelRegister.register("Liner")
class LinerMeasureModel(BasicMeasure):
    para_list = ["px", "vx", "py", "vy", "pz", "vz"]

    def __init__(self, XDim, StationLocation=None):
        super(LinerMeasureModel, self).__init__(XDim, StationLocation)

        MDim = len(self.para_list)
        if MDim == 6:
            self.H = np.eye(XDim)[[0, 1, 3, 4, 6, 7], :]  # Default1
        else:
            self.H = np.eye(XDim)[:MDim, :]  # Default2

    def MultMeasure(self, AbsoluteStateVectors):
        RelativeStateVectors = AbsoluteStateVectors.T - self.StationLocation
        Z = self.H.dot(RelativeStateVectors.T)
        return Z


@MeasureModelRegister.register("Radar3D")
class RadarSensorMeasureModel(BasicMeasure):
    para_list = ['rRadial', 'vRadial', 'Beta', 'Zeta']

    def __init__(self, XDim, StationLocation=None):
        super(RadarSensorMeasureModel, self).__init__(XDim, StationLocation)

    def SingleMeasure(self, StateVector):
        RelativeStateVectors = StateVector - self.StationLocation  # 此径向距离是从测量点算的
        vecR = RelativeStateVectors[[0, 3, 6]]
        vecV = RelativeStateVectors[[1, 4, 7]]
        vecA = RelativeStateVectors[[2, 5, 8]]
        # vecLoc = self.location
        rRadial = np.linalg.norm(vecR, axis=0)
        vRadial = np.sum(vecR * vecV, axis=0) / rRadial
        Beta = np.arctan2(*vecR[[1, 0]])
        Zeta = np.arctan2(vecR[2], np.linalg.norm(vecR[[0, 1]], axis=0))

        Z = np.r_[rRadial, vRadial, Beta, Zeta]
        return Z

    def MultMeasure(self, AbsoluteStateVectors, *args):
        RelativeStateVectors = AbsoluteStateVectors.T - self.StationLocation  # 此径向距离是从测量点算的
        RelativeStateVectors = RelativeStateVectors.T
        vecR = RelativeStateVectors[[0, 3, 6]]
        vecV = RelativeStateVectors[[1, 4, 7]]
        vecA = RelativeStateVectors[[2, 5, 8]]
        # vecLoc = self.location
        rRadial = np.linalg.norm(vecR, axis=0)
        vRadial = np.sum(vecR * vecV, axis=0) / rRadial
        Beta = np.arctan2(*vecR[[1, 0]])
        Zeta = np.arctan2(vecR[2], np.linalg.norm(vecR[[0, 1]], axis=0))

        Z = np.c_[rRadial, vRadial, Beta, Zeta].T
        return Z

    def MeasureJacobi(self, AbsoluteStateVector):
        # dimFlag = AbsoluteStateVectors.ndim

        AbsoluteStateVectors = AbsoluteStateVector
        hOut = np.zeros([self.MeasurementDim, self.StateDim])

        RelativeStateVectors = AbsoluteStateVectors - self.StationLocation
        # hOut = np.zeros([4, 9])
        # hOut = super(StandstillRadarSensorModel, self).measureJacobi(X)
        vecR = RelativeStateVectors[[0, 3, 6]]
        vecV = RelativeStateVectors[[1, 4, 7]]
        # r = np.sqrt(StateVec[0] ** 2 + StateVec[3] ** 2)
        r = np.linalg.norm(vecR, axis=0)
        r_xy = np.linalg.norm(vecR[:-1], axis=0)

        # 对r求导数
        hOut[0, [0, 3, 6]] = vecR / r

        # 对rdot求导数
        hOut[1, [0, 3, 6]] = vecV / r - vecR * np.sum(vecR * vecV, axis=0) / r ** 3
        hOut[1, [1, 4, 7]] = vecR / r

        # 对theta求导数
        hOut[2, 0] = -vecR[1] / r_xy ** 2
        hOut[2, 3] = vecR[0] / r_xy ** 2

        x = vecR[0]
        y = vecR[1]
        z = vecR[2]
        hOut[3, [0, 3, 6]] = np.array([x * z, y * z, x ** 2 + y ** 2]) / (r ** 2 * r_xy)
        return hOut


    # def MeasureJacobi(self, AbsoluteStateVectors):
    #     dimFlag = AbsoluteStateVectors.ndim
    #     if dimFlag == 1:
    #         AbsoluteStateVectors = AbsoluteStateVectors[:, None]
    #         hOut = np.zeros([self.MeasurementDim, self.StateDim, 1])
    #     elif dimFlag  == 2:
    #         hOut = np.zeros([self.MeasurementDim, self.StateDim, AbsoluteStateVectors.shape[1]])
    #     else:
    #         """
    #         没有做更高维度的了
    #         """
    #         return None
    #     RelativeStateVectors = AbsoluteStateVectors - self.StationLocation[:, None]
    #     # hOut = np.zeros([4, 9])
    #     # hOut = super(StandstillRadarSensorModel, self).measureJacobi(X)
    #     vecR = RelativeStateVectors[[0, 3, 6]]
    #     vecV = RelativeStateVectors[[1, 4, 7]]
    #     # r = np.sqrt(StateVec[0] ** 2 + StateVec[3] ** 2)
    #     r = np.linalg.norm(vecR, axis=0)
    #     r_xy = np.linalg.norm(vecR[:-1], axis=0)
    #
    #     # 对r求导数
    #     hOut[0, [0, 3, 6], :] = vecR / r
    #
    #     # 对rdot求导数
    #     hOut[1, [0, 3, 6], :] = vecV / r - vecR * np.sum(vecR * vecV, axis=0) / r ** 3
    #     hOut[1, [1, 4, 7], :] = vecR / r
    #
    #     # 对theta求导数
    #     hOut[2, 0, :] = -vecR[1] / r_xy ** 2
    #     hOut[2, 3, :] = vecR[0] / r_xy ** 2
    #
    #     x = vecR[0]
    #     y = vecR[1]
    #     z = vecR[2]
    #     hOut[3, [0, 3, 6], :] = np.array([x * z, y * z, x ** 2 + y ** 2]) / (r ** 2 * r_xy)
    #     if dimFlag == 1:
    #         return hOut.transpose([2, 0, 1])[0]
    #     else:
    #         return hOut.transpose([2, 0, 1])

class MeasureFactory(BasicFactory):
    def __init__(self):
        super(MeasureFactory, self).__init__()
        self.service_dict = MeasureModelRegister
