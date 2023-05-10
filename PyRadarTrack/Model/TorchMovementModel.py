#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :TorchMovementModel.py
# @Time      :2023/5/10 1:42 PM
# @Author    :Oliver

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from PyRadarTrack.Core.Register import Register
from PyRadarTrack.Core.BasicFactory import BasicFactory
from PyRadarTrack.Core.BasicObject import BasicObject
TorchMovementRegister = Register()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseMovementModel(torch.nn.Module, BasicObject):
    """
    运动模型最基础的功能就是：根据本时刻的状态向量推演下一时刻目标的状态向量
    """
    def __init__(self, XDim=9, QDim=9, *args, **kwargs):
        # super(BaseMovementModel, self).__init__()
        # super(torch.nn.Module,self).__init__()
        super().__init__(*args, **kwargs)
        self.XDim = XDim
        self.QDim = QDim

        self.F = torch.eye(self.XDim, device=device)
        self.Q = torch.eye(self.QDim, device=device)
        self.G = torch.zeros([self.XDim, self.QDim], device=device)
        if self.XDim == self.QDim:
            self.setGArray(torch.eye(self.XDim, device=device))
        self.dt = None
        self.QSigma = None
        if device.type =="cpu":
            self.M = MultivariateNormal(torch.zeros(self.QDim), self.Q)     # 这种实现似乎不容易在GPU上跑
            self.forward = self.nextstepCpu
        else:
            self.sqrtQ = torch.linalg.cholesky(self.Q)
            self.forward = self.nextstepGeneral

    def setGArray(self, G):
        self.G[:] = G

    def setQ(self,Q):
        self.Q[:] = Q
        if device.type == "cpu":
            self.M = MultivariateNormal(torch.zeros(self.QDim), self.Q)
        else:
            self.sqrtQ = torch.linalg.cholesky(self.Q)


    def nextstepCpu(self, X: torch.tensor):
        X = X.unsqueeze(-1)
        NextX = self.F @ X
        Noise = self.G @ self.M.sample(X.shape[:-2]).unsqueeze(-1)
        return (Noise + NextX).squeeze(-1)

    def nextstepGeneral(self,X:torch.tensor):
        X = X.unsqueeze(-1)
        NextX = self.F @ X
        # Noise = self.G @ self.sqrtQ @ torch.randn(list(X.shape[:-2])+[self.QDim]).unsqueeze(-1)
        Noise = self.G @ self.sqrtQ @ torch.randn(X.shape)  # 注意这种写法只有在XDim==QDim时才能成立,这样写是因为上面那个太蠢了
        return (Noise + NextX).squeeze(-1)
    def EnsuringXStraight(self, X):
        return X


import numpy as np
@TorchMovementRegister.register
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


@TorchMovementRegister.register
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


@TorchMovementRegister.register
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
        self.service_dict = TorchMovementRegister

if __name__ == '__main__':

    test = BaseMovementModel()

    d = test(torch.zeros(4,9))
    d2 = test(torch.zeros(9))
    test.sqrtQ = torch.linalg.cholesky(test.Q)
    d4 = test.nextstepGeneral(torch.zeros(4,9))
    d3 = test.nextstepGeneral(torch.zeros(9))