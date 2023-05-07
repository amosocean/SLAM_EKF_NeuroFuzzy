#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Target.py
# @Time      :2023/1/4 11:56 PM
# @Author    :Kinddle
"""
Target Generator
用于根据特性生成目标轨迹-通过较为简单的指令来生成可用的目标轨迹
读取已有轨迹-需要统一接口，先不做
需要先有雷达系统才能产生
"""
import numpy as np
import pandas as pd

from ..Core import *
# from.SystemCfg import SystemCfg
from ..Interface import Message

TargetRegister = Register()


class BaseTarget(BasicObjectWithCfg):
    para_list = ['px', 'vx', 'ax', 'py', 'vy', 'ay', 'pz', 'vz', 'az', 'timestamp']
    ConfigClass = SystemCfg

    def __init__(self, parents=None):
        self.TargetRecorder = Recorder(self.para_list)
        super(BaseTarget, self).__init__(parents)

    def loadSystemCfg(self, SystemCfg_):
        super(BaseTarget, self).loadSystemCfg(SystemCfg_)
        self.TargetRecorder.setBufferLen(self.SystemCfg['SimulationTimeTicks'])
        # tmp = self.TargetRecorder.clear()
        # self.TargetRecorder = Recorder(self.para_list, self.SystemCfg['SimulationTimeTicks'])
        # self.TargetRecorder.load(tmp)

    # region ---[.]get set
    def get_real_data_all(self):
        return self.TargetRecorder.get_data_all()

    def get_real_data_index(self, index):
        return self.TargetRecorder.get_data(index)

    def data_get_now(self):
        return self.TargetRecorder.get_data()

    def data_finish(self):
        return self

    def data_clear(self):
        return self.TargetRecorder.clear()

    # endregion


@TargetRegister.register("TFK")
class TargetFromKeyframe(BaseTarget):
    def __init__(self, parents=None, nowX=None):
        super(TargetFromKeyframe, self).__init__(parents)
        self.nowX = nowX
        self.timestamp = 0
        self.X0 = None

    def reset(self, X0=None):
        if X0 is None:
            X0 = self.X0
        rtn = self.TargetRecorder.clear()
        self.timestamp = 0
        self._step(X0)
        return rtn

    def step(self, X):
        if self.nowX is None:
            self.X0 = X
        return self._step(X)

    def _step(self, X):
        self.nowX = X
        self.TargetRecorder.step(np.r_[X, self.timestamp])
        self.timestamp += self.Ts
        return self

    def run_Model(self, Model, runtime):
        """
        使用这种方法的前提是有初始值（nowX非空）
        """
        for i in range(runtime):
            self._step(Model.nextstep(self.nowX))
        return self

    # def data_by_Model(self, Model, runtime):
    #     for i in range(runtime):
    #         self.data_nextTick_set(Model.nextstep(self.data_get_now()))
    #     return self
    #
    # def data_init(self, x0, nowTime=0):
    #     if not self.SystemCfg:
    #         Message.warning("需要先定义系统参数才能构建轨迹，详见SimulationBox类")
    #         return None
    #
    #     self.buffer_data = np.zeros([self.SystemCfg['StateDim'], self.SystemCfg['SimulationTimeTicks']])
    #     self.buffer_timestamp = np.zeros([self.SystemCfg['SimulationTimeTicks']]) - 1
    #
    #     self.buffer_data[:, 0] = x0
    #     self.nowTick = 0
    #     self.buffer_timestamp[0] = nowTime
    #     return self
    #
    # def data_finish(self):
    #     save_data = pd.DataFrame(np.c_[self.buffer_data[:, :self.nowTick + 1].transpose(),
    #                                    self.buffer_timestamp[:self.nowTick + 1, None]],
    #                              columns=self.para_list)
    #     self.real_data = self.real_data.append(save_data, ignore_index=True)
    #     self.nowTick = -1  # 存完的数据
    #     return self
    #
    # def data_overflow(self, X):
    #     self.data_finish()
    #
    #     # save_data = pd.DataFrame(np.c_[self.buffer_data[:, :self.nowTick + 1].transpose(),
    #     # self.buffer_timestamp[:self.nowTick + 1, None]], columns=self.para_list)
    #     # self.real_data = self.real_data.append(save_data,ignore_index=True) self.nowTick = -1
    #
    #     Message.warning("存储数据超出预定义长度，尝试扩展长度")
    #     now_time = self.buffer_timestamp[-1] + self.SystemCfg['Ts']
    #
    #     self.data_init(X, now_time)
    #
    #     # self.buffer_data = np.zeros([self.SystemCfg['StateDim'], self.SystemCfg['SimulationTimeTicks']])
    #     # self.buffer_timestamp = np.zeros([self.SystemCfg['SimulationTimeTicks']]) - 1
    #     # self.buffer_data[:, 0] = X
    #     # self.nowTick = 0
    #     # self.buffer_timestamp[0] = now_time + self.SystemCfg['Ts']
    #     return self
    #
    # def data_get_now(self):
    #     return self.buffer_data[:, self.nowTick]
    #
    # def data_nowTick_add(self, addX):
    #     self.buffer_data[:, self.nowTick] += addX
    #
    # def data_nextTick_set(self, X):
    #     self.nowTick += 1
    #     if self.nowTick >= self.SystemCfg["SimulationTimeTicks"]:
    #         self.data_overflow(X)
    #     else:
    #         self.buffer_timestamp[self.nowTick] = self.buffer_timestamp[self.nowTick - 1] + self.SystemCfg["Ts"]
    #         self.buffer_data[:, self.nowTick] = X
    #     return self


class TargetFactory(BasicFactory):
    def __init__(self):
        super(TargetFactory, self).__init__()
        self.service_dict = TargetRegister
