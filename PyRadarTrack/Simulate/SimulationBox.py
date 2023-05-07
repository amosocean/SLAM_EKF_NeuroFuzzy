#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :SimulationBox.py
# @Time      :2023/1/6 10:01 PM
# @Author    :Kinddle
"""
仿真盒
整个仿真过程的顶层，可以定义一些环境整体的参数。
在仿真盒中可以加载多个轨迹和
"""
from ..Core import *

# from .SystemCfg import SystemCfg

SimulationBoxRegister = Register()
_Fs = 1
_DefaultSystemConfig = {"C": 3e8,
                        "Fs": _Fs,
                        "Ts": 1 / _Fs,
                        "StateDim": 9,
                        "MeasurementDim": 4,
                        "SimulationTimeTicks": 500}


class BasicSimulationBox(BasicObjectWithCfg):
    ConfigClass = SystemCfg

    def __init__(self, SystemConfig=None):
        # if SystemConfig is None:
        SystemConfig = dict(**_DefaultSystemConfig, **{} if SystemConfig is None else SystemConfig)
        super(BasicSimulationBox, self).__init__(SystemConfig)
        self._Targets = []
        self._Sensors = []

    def SystemCfgUpdate(self, update_dic):
        self.SystemCfg.setConfig(update_dic)
        self.SystemCfg.sync(self)
        # self.SystemCfg.update(update_dic)

    def addTarget(self, Target):
        # 以防万一有轨迹没有结束
        self._Targets.append(Target)

    def addSensor(self, Sensor):
        # 备忘：
        # Sensor的时序需要是Ts的整数倍
        # 以防万一有轨迹没有结束
        self._Sensors.append(Sensor)

    def step(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        TargetsData = []
        SensorsData = []
        for Target in self._Targets:
            TargetsData.append(Target.reset())
        for Sensor in self._Sensors:
            SensorsData.append(Sensor.reset())

        return TargetsData, SensorsData


@SimulationBoxRegister.register("SimBoxLiner")
class SimulationBox(BasicSimulationBox):

    def __init__(self):
        super(SimulationBox, self).__init__()


@SimulationBoxRegister.register("SimBoxRadar")
class SimulationBoxFromConf(BasicSimulationBox):
    def __init__(self):
        super(SimulationBoxFromConf, self).__init__()
        pass

@SimulationBoxRegister.register("SLAM")
class SimulationBoxFromConf(BasicSimulationBox):
    def __init__(self):
        super(SimulationBoxFromConf, self).__init__()
        pass

class SimulationBoxFactory(BasicFactory):
    def __init__(self):
        super(SimulationBoxFactory, self).__init__()
        self.service_dict = SimulationBoxRegister
