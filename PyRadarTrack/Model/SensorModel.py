#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :SensorModel.py.py
# @Time      :2023/1/11 1:49 PM
# @Author    :Kinddle

from ..Core import *
from .MeasureNoiseModel import *
from .MeasureModel import *
from .FilterModel import *

SensorModelRegister = Register()


# H_default = np.eye()

class BasicSensorModel(BasicObjectWithCfg):
    ConfigClass = SystemCfg
    ParaX_list = [] + ["timestamp"]
    ParaM_list = [] + ["timestamp"]

    def __init__(self, StationLocation, parents=None):
        self.StateDim = len(self.ParaX_list) - 1
        self.MeasurementDim = len(self.ParaM_list) - 1

        self.StationLocation = StationLocation
        self.timestamp = 0
        self.Ts = None

        self.PureMeasureRecorder = Recorder(self.ParaM_list)
        self.MixMeasureRecorder = Recorder(self.ParaM_list)
        self.EstimateRecorder = Recorder(self.ParaX_list)

        super(BasicSensorModel, self).__init__(parents)

        self._MeasureModel = None
        self._MeasureNoiseModel = None
        self._FilterModel = None

    def record_packet(self, Est, PureM, MixM):
        self.EstimateRecorder.step(np.r_[Est, self.timestamp])
        self.PureMeasureRecorder.step(np.r_[PureM, self.timestamp])
        self.MixMeasureRecorder.step(np.r_[MixM, self.timestamp])
        return self

    def reset(self):
        self.timestamp = 0
        rtn = self.getRecorderData()
        self.PureMeasureRecorder = Recorder(self.ParaM_list)
        self.MixMeasureRecorder = Recorder(self.ParaM_list)
        self.EstimateRecorder = Recorder(self.ParaX_list)
        self._FilterModel.reset()
        return rtn

    def _Noise(self, *args, **kwargs):
        return self._MeasureNoiseModel.Noise(*args, **kwargs)

    def _Measure(self, X, *args, **kwargs):
        return self._MeasureModel.MultMeasure(X, *args, **kwargs)

    # region ----[.]get, set等支持函数
    def getMeasureModel(self):
        return self._MeasureModel

    def getMeasureNoiseModel(self):
        return self._MeasureNoiseModel

    def getFilterModel(self):
        return self._FilterModel

    def setMeasureModel(self, Model):
        self._MeasureModel = Model
        return self

    def setMeasureNoiseModel(self, Model):
        self._MeasureNoiseModel = Model
        return self

    def setFilterModel(self, Model):
        self._FilterModel = Model
        return self

    def getRecorders(self):
        return self.EstimateRecorder, self.PureMeasureRecorder, self.MixMeasureRecorder

    def getRecorderData(self):
        rtn = {"EstimateRecorder": self.EstimateRecorder.get_data_all(),
               "PureMeasureRecorder": self.PureMeasureRecorder.get_data_all(),
               "MixMeasureRecorder": self.MixMeasureRecorder.get_data_all(), }
        return rtn
    # endregion


@SensorModelRegister.register("Liner_KF")
class LinerSensorModel(BasicSensorModel):
    ParaX_list = ["px", "vx", "ax", "py", "vy", "ay", "pz", "vz", "az"] + ['timestamp']
    ParaM_list = ["px", "vx", "py", "vy", "pz", "vz"] + ['timestamp']

    def __init__(self, StationLocation, parents=None):
        super(LinerSensorModel, self).__init__(StationLocation, parents)
        # DefaultH = np.eye(self.StateDim)[[0, 1, 3, 4, 6, 7], :]
        self._DEFAULT_model_config(StationLocation)

    def _DEFAULT_model_config(self, StationLocation):
        self.setMeasureModel(LinerMeasureModel(self.StateDim, StationLocation))
        self.setMeasureNoiseModel(
            LinerMeasureNoiseModel(self.StateDim, self.MeasurementDim).setSqrtR(np.diag([10, 2] * 3)))
        self.setFilterModel(BasicKFModel(self.ParaX_list, self.ParaM_list))
        self.H = self._MeasureModel.H
        self._DEFAULT_model_config = None     # 重载到一个空函数来保证这个函数只会被使用一次

    def step(self, X, Record=True):
        Z = self._Measure(X)
        Z_n = self._Noise()
        Xkf = self._FilterModel.step(Z + Z_n)
        if Record:
            self.timestamp += self.Ts
            self.record_packet(Xkf, Z, Z+Z_n)

        return Xkf

    def run(self, RealData):
        Xkf = np.zeros(RealData.shape)
        for t in range((RealData.shape[1])):
            X = RealData[:, t]
            Xkf[:, t] = self.step(X)
        return Xkf


@SensorModelRegister.register("Radar_A")
class RadarSensorModelA(BasicSensorModel):
    """
    并非开盒即用..
    仅使用EKF方法的模型
    """
    ParaX_list = ["px", "vx", "ax", "py", "vy", "ay", "pz", "vz", "az"] + ['timestamp']
    ParaM_list = ['rRadial', 'vRadial', 'Beta', 'Zeta'] + ['timestamp']

    def __init__(self, StationLocation, parents=None):
        super(RadarSensorModelA, self).__init__(StationLocation, parents)
        self._DEFAULT_model_config(StationLocation,parents)

    def _DEFAULT_model_config(self, StationLocation, parents):
        self.setMeasureModel(RadarSensorMeasureModel(self.StateDim, StationLocation))
        self.setMeasureNoiseModel(PSModelGPLFM(self.StateDim, self.MeasurementDim).setPara(C=self.C))
        self.setFilterModel(BasicEKFModel(XPara_list=self.ParaX_list, MPara_list=self.ParaM_list, parents=parents)
                            .setJacobiFun(self.getMeasureModel().MeasureJacobi)
                            .setMeasureFun(self.getMeasureModel().MultMeasure))
        self._DEFAULT_model_config = None     # 重载到一个空函数来保证这个函数只会被使用一次

    def reset(self):
        self.timestamp = 0
        rtn = self.getRecorderData()
        self.PureMeasureRecorder = Recorder(self.ParaM_list)
        self.MixMeasureRecorder = Recorder(self.ParaM_list)
        self.EstimateRecorder = Recorder(self.ParaX_list)
        self._FilterModel.reset()

        Z = self._Measure(self._FilterModel.X0).T[0]

        self.record_packet(self._FilterModel.X0, Z, Z)
        return rtn

    def SyncMovementModel(self, MovementModel, X0, P0, R, Record=True):
        Tmp = self.getFilterModel()
        Tmp.ModelInit(MovementModel.F, Tmp.H_Func, MovementModel.Q, R)
        Tmp.DataInit(X0, P0)
        if Record:
            Z = self._Measure(X0).T[0]
            self.record_packet(X0, Z, Z)
        self.setFilterModel(Tmp)
        return self

    def step(self, X, lmd, b, Record=True):
        Z = self._Measure(X).T[0]
        Z_n = self._Noise(R_measure=Z[0], lmd=lmd, b=b)
        # Z_n = 0
        # self._Noise(R_measure=(Z+Z_n)[0], lmd=lmd, b=b)  # 实际情况中只能通过测量获得的R，调用Noise是为了更新新的R矩阵
        self._FilterModel.setR(self._MeasureNoiseModel.getRWithPara({"R_measure": (Z + Z_n)[0],
                                                                     "lmd": lmd,
                                                                     "b": b}))  # 很关键的步骤 同步滤波器和测量噪声的R矩阵
        Xkf = self._FilterModel.step(Z + Z_n, R_measure=(Z + Z_n)[0], lmd=lmd, b=b)
        if Record:
            self.timestamp += self.Ts
            self.record_packet(Xkf, Z, Z+Z_n)
        return Xkf

    def run(self, RealData, lmdData, bData):
        Xkf = np.zeros(RealData.shape)
        for t in range((RealData.shape[1])):
            X = RealData[:, t]
            lmd = lmdData[t]
            b = bData[t]
            Xkf[:, t] = self.step(X, lmd, b)
        return Xkf


@SensorModelRegister.register("Radar_B")
class RadarSensorModelB(BasicSensorModel):
    """
    并非开盒即用
    """
    ParaX_list = ["px", "vx", "ax", "py", "vy", "ay", "pz", "vz", "az"] + ['timestamp']
    ParaM_list = ['rRadial', 'vRadial', 'Beta', 'Zeta'] + ['timestamp']

    def __init__(self, StationLocation, parents=None):
        super(RadarSensorModelB, self).__init__(StationLocation, parents)
        self._DEFAULT_model_config(StationLocation, parents)

    def _DEFAULT_model_config(self, StationLocation, parents):
        self.setMeasureModel(RadarSensorMeasureModel(self.StateDim, StationLocation))
        self.setMeasureNoiseModel(PSModelGPLFM(self.StateDim, self.MeasurementDim).setPara(C=self.C))
        self.setFilterModel(BasicEKFModel(XPara_list=self.ParaX_list, MPara_list=self.ParaM_list, parents=parents)
                            .setJacobiFun(self.getMeasureModel().MeasureJacobi)
                            .setMeasureFun(self.getMeasureModel().MultMeasure))
        self._DEFAULT_model_config = None  # 重载到一个空函数来保证这个函数只会被使用一次

    def reset(self):
        self.timestamp = 0
        rtn = self.getRecorderData()
        self.PureMeasureRecorder = Recorder(self.ParaM_list)
        self.MixMeasureRecorder = Recorder(self.ParaM_list)
        self.EstimateRecorder = Recorder(self.ParaX_list)
        self._FilterModel.reset()

        Z = self._Measure(self._FilterModel.X0).T[0]
        self.record_packet(self._FilterModel.X0, Z, Z)
        return rtn

    def SyncMovementModel(self, MovementModel, X0, P0, R, Record=True):
        Tmp = self.getFilterModel()
        Tmp.ModelInit(MovementModel.F, Tmp.H_Func, MovementModel.Q, R)
        Tmp.DataInit(X0, P0)
        if Record:
            Z = self._Measure(X0).T[0]
            self.record_packet(X0, Z, Z)
        self.setFilterModel(Tmp)
        return self

    def step(self, X, lmd, b, Record=True):
        Z = self._Measure(X).T[0]
        Z_n = self._Noise(R_measure=Z[0], lmd=lmd, b=b)

        # self._Noise(R_measure=(Z + Z_n)[0], lmd=lmd, b=b)  # 实际情况中只能通过测量获得的R，调用Noise是为了更新新的R矩阵
        self._FilterModel.setR(self._MeasureNoiseModel.getRWithPara({"R_measure": (Z + Z_n)[0],
                                                                     "lmd": lmd,
                                                                     "b": b}))  # 很关键的步骤 同步滤波器和测量噪声的R矩阵
        # self._FilterModel.setR(self._MeasureNoiseModel.R)  # 很关键的步骤 同步滤波器和测量噪声的R矩阵
        Xkf = self._FilterModel.step(Z + Z_n)
        if Record:
            self.timestamp += self.Ts
            self.record_packet(Xkf, Z, Z+Z_n)
        return Xkf

    def run(self, RealData, lmdData, bData):
        Xkf = np.zeros(RealData.shape)
        for t in range((RealData.shape[1])):
            X = RealData[:, t]
            lmd = lmdData[t]
            b = bData[t]
            Xkf[:, t] = self.step(X, lmd, b)
        return Xkf


class SensorModelFactory(BasicFactory):
    def __init__(self):
        super(SensorModelFactory, self).__init__()
        self.service_dict = SensorModelRegister
