#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MovementModel.py
# @Time      :2023/1/5 6:56 PM
# @Author    :Kinddle
"""
运动模型的设计
需要有状态转移矩阵F
和过程噪声协方差Q
大概还有噪音驱动方程G
都包括三维
"""
import numpy as np

from ..Core import *

# from ..Interface import *

MovementRegister = Register()


class BaseMovementModel(BasicObject):
    """
    运动模型最基础的功能就是：根据本时刻的状态向量推演下一时刻目标的状态向量
    """
    XDim = 9
    QDim = 9

    def __init__(self, *args, **kwargs):
        super(BaseMovementModel, self).__init__()
        self.F = np.eye(self.XDim)
        self.Q = np.eye(self.QDim)
        self.G = np.zeros([self.XDim, self.QDim])
        if self.XDim == self.QDim:
            self.setGArray(np.eye(self.XDim))
        self.dt = None
        self.QSigma = None

    def setGArray(self, G):
        self.G[:] = G

    def nextstep(self, X: np.array):
        return self.F.dot(X) + self.G.dot(sqrtm(self.Q).dot(np.random.randn(self.QDim)))

    def EnsuringXStraight(self, X):
        return X


@MovementRegister.register
class CVModel(BaseMovementModel):

    def __init__(self, dt, Sigma, *args, **kwargs):
        super(CVModel, self).__init__()
        self.setFQ(dt, Sigma)

    def setFQ(self, dt, Sigma):
        self.dt = dt
        self.QSigma = Sigma
        self.F = np.kron(np.eye(3), np.array([[1, dt, 0], [0, 1, 0], [0, 0, 0]]))
        self.Q = np.kron(np.eye(3),
                         np.array([[dt ** 3 / 3, dt ** 2 / 2, 0], [dt ** 2 / 2, dt, 0], [0, 0, 0]])) * self.QSigma
        # 此乃平面
        # self.F = np.array([[1, dt, 0, 0, 0, 0],
        #                    [0, 1, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, 1, dt, 0],
        #                    [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 0], ])
        # self.Q = np.array([[dt ** 3 / 3, dt ** 2 / 2, 0, 0, 0, 0],
        #                    [dt ** 2 / 2, dt, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, dt ** 3 / 3, dt ** 2 / 2, 0],
        #                    [0, 0, 0, dt ** 2 / 2, dt, 0],
        #                    [0, 0, 0, 0, 0, 0], ]) * self.Sigma


@MovementRegister.register
class CAModel(BaseMovementModel):

    def __init__(self, dt, Sigma, *args, **kwargs):
        super(CAModel, self).__init__()
        self.setFQ(dt, Sigma)

    def setFQ(self, dt, Sigma):
        self.dt = dt
        self.QSigma = Sigma
        self.F = np.kron(np.eye(3), np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]]))
        # self.Q = np.kron(np.eye(3),
        #                  np.array([[dt ** 3 / 3, dt ** 2 / 2, 0], [dt ** 2 / 2, dt, 0], [0, 0, 0]])) * self.QSigma
        self.Q = np.kron(np.eye(3), np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                                              [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                                              [dt ** 3 / 6, dt ** 2 / 2, dt]])) * self.QSigma
        # self.F = np.array([[1, dt, dt ** 2 / 2, 0, 0, 0],
        #                    [0, 1, dt, 0, 0, 0],
        #                    [0, 0, 1, 0, 0, 0],
        #                    [0, 0, 0, 1, dt, dt ** 2 / 2],
        #                    [0, 0, 0, 0, 1, dt],
        #                    [0, 0, 0, 0, 0, 1], ])
        # self.Q = np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6, 0, 0, 0],
        #                    [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2, 0, 0, 0],
        #                    [dt ** 3 / 6, dt ** 2 / 2, dt, 0, 0, 0],
        #                    [0, 0, 0, dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
        #                    [0, 0, 0, dt ** 5 / 20, dt ** 3 / 3, dt ** 2 / 2],
        #                    [0, 0, 0, dt ** 3 / 6, dt ** 2 / 2, dt], ]) * self.QSigma

    def EnsuringXStraight(self, X):
        # 将加速度改变到速度所对应的方向
        # vecR = X[[0, 3, 6]]
        X = X.copy()
        vecV = X[[1, 4, 7]]
        vecA = X[[2, 5, 8]]
        normA = np.linalg.norm(vecA)
        DirectV = vecV / np.linalg.norm(vecV)
        newA = DirectV * normA
        X[[2, 5, 8]] = newA
        return X


@MovementRegister.register
class CTxyModel(BaseMovementModel):
    """
    只能处理绕着xy平面旋转的情况且沿z轴
     未来有更好的模型请写在后面谢谢。
    """

    def __init__(self, dt, Sigma, w, *args, **kwargs):
        super(CTxyModel, self).__init__()
        self.w = None
        self.setFQ(dt, Sigma, w)

    def setFQ(self, dt, Sigma, w):
        self.dt = dt
        self.QSigma = Sigma
        self.w = w
        # self.F = np.array([[1, np.sin(w * dt) / w, 0, 0, -(1 - np.cos(w * dt)) / w, 0, 0, 0, 0],
        #                    [0, np.cos(w * dt), 0, 0, -np.sin(w * dt), 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [0, (1 - np.cos(w * dt)) / w, 0, 1, np.sin(w * dt) / w, 0, 0, 0, 0, 0],
        #                    [0, np.sin(w * dt), 0, 0, np.cos(w * dt), 0, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0, 0, 1, dt, 0],
        #                    [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
        #                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],])
        self.F = np.zeros([self.XDim, self.XDim])
        Fw = lambda w: np.array([[1, np.sin(w * dt) / w, 0, 0, -(1 - np.cos(w * dt)) / w, 0],
                                 [0, np.cos(w * dt), 0, 0, -np.sin(w * dt), 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, (1 - np.cos(w * dt)) / w, 0, 1, np.sin(w * dt) / w, 0],
                                 [0, np.sin(w * dt), 0, 0, np.cos(w * dt), 0],
                                 [0, 0, 0, 0, 0, 0], ]) if w != 0 \
            else np.kron(np.eye(3), np.array([[1, dt, 0], [0, 1, 0], [0, 0, 0]]))
        self.F[0:6, 0:6] += Fw(w)
        self.F[6:9, 6:9] += np.array(np.array([[1, dt, 0], [0, 1, 0], [0, 0, 0]]))

        self.Q = np.kron(np.eye(3),
                         np.array([[dt ** 3 / 3, dt ** 2 / 2, 0], [dt ** 2 / 2, dt, 0], [0, 0, 0]])) * self.QSigma


# @MovementRegister.register
class CTModelErr(BaseMovementModel):
    """
    一个错误的模型 但我先不删
    """

    def __init__(self, dt, Sigma, wx, wy, wz):
        super(CTModelErr, self).__init__()
        self.wx = None
        self.wy = None
        self.wz = None
        self.setFQ(dt, Sigma, wx, wy, wz)

    def setFQ(self, dt, Sigma, wx, wy, wz):
        self.dt = dt
        self.QSigma = Sigma
        self.wx = wx
        self.wy = wy
        self.wz = wz
        Fw = lambda w: np.array([[1, np.sin(w * dt) / w, 0, 0, -(1 - np.cos(w * dt)) / w, 0],
                                 [0, np.cos(w * dt), 0, 0, -np.sin(w * dt), 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, (1 - np.cos(w * dt)) / w, 0, 1, np.sin(w * dt) / w, 0],
                                 [0, np.sin(w * dt), 0, 0, np.cos(w * dt), 0],
                                 [0, 0, 0, 0, 0, 0], ]) if w != 0 \
            else np.kron(np.eye(3), np.array([[1, dt, 0], [0, 1, 0], [0, 0, 0]]))

        self.F = np.zeros([self.XDim, self.XDim])
        self.F[0:6, 0:6] += Fw(self.wz)
        self.F[[[i] for i in [0, 1, 2, 6, 7, 8]], [0, 1, 2, 6, 7, 8, ]] += Fw(self.wy)
        self.F[3:9, 3:9] += Fw(self.wx)

        # self.Q = np.array([[dt ** 3 / 3, dt ** 2 / 2, 0, 0, 0, 0],
        #                    [dt ** 2 / 2, dt, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, dt ** 3 / 3, dt ** 2 / 2, 0],
        #                    [0, 0, 0, dt ** 2 / 2, dt, 0],
        #                    [0, 0, 0, 0, 0, 0], ]) * self.Sigma
        self.Q = np.kron(np.eye(3),
                         np.array([[dt ** 3 / 3, dt ** 2 / 2, 0], [dt ** 2 / 2, dt, 0], [0, 0, 0]])) * self.QSigma


class MovementModelFactory(BasicFactory):

    def __init__(self):
        super(MovementModelFactory, self).__init__()
        self.service_dict = MovementRegister
