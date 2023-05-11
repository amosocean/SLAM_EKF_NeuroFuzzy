#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Track_Generate.py
# @Time      :2023/5/7 6:52 PM
# @Author    :Oliver
import copy
import torch
import numpy as np
from PyRadarTrack.Model import *
from PyRadarTrack.Simulate import *
from PyRadarTrack.Model.FilterModel import IMMFilterModel, BasicEKFModel
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)


class Basic_Track_Dataset_Generate(torch.utils.data.Dataset):

    Scale_vector = np.array([3000, 10, 1e-3] * 3)
    MovementModelNameMap = {"CV": "CVModel",
                            "CT": "CTxyModel",
                            "CA": "CAModel"}

    def __init__(self, Simulate_frame, dt=0.1, Sigma=0.01, xWin=5, yWin=5, seed=None,
                 UsedModel=None,
                 Flag_withTime=False):
        super().__init__()
        self.xWin = xWin
        self.yWin = yWin
        self.seed = seed
        self.Simulate_frame = Simulate_frame
        self.Flag_withTime = Flag_withTime
        # self.Flag_transpose = Flag_transpose

        # region --准备运动模型
        self.SB = SimulationBox()
        self.SB.SystemCfgUpdate({"Ts": dt,
                                 "QSigma": Sigma,
                                 "SimulationTimeTicks": Simulate_frame})
        self.MMF = MovementModelFactory()
        self.MovementModels =[]
        if UsedModel is None:
            UsedModel = self.MovementModelNameMap.keys()
        if "CV" in UsedModel:
            self.MovementModels.append(self.MMF.create('CVModel')(dt, Sigma))
        if "CT" in UsedModel:
            self.MovementModels.append(self.MMF.create('CTxyModel')(dt, Sigma, -0.35))
        if "CA" in UsedModel:
            self.MovementModels.append( self.MMF.create('CAModel')(dt, Sigma))

        # endregion

        # region-- 初始化中间变量
        self.pure_track = None
        self.noisy_track=None
        # endregion

        self.gen_randomTrack()
        self.add_noise()

    def gen_randomTrack(self, init_point=None, div_num=10):
        if self.seed:
            np.random.seed(self.seed)
        X0 = init_point if init_point else np.random.randn(9) * self.Scale_vector
        Track = TargetFromKeyframe(self.SB)
        Track.step(X0)
        ShiftTime = np.r_[0, np.sort(np.random.choice(np.arange(self.Simulate_frame - 1), div_num)), self.Simulate_frame - 1]
        StayTime = ShiftTime[1:] - ShiftTime[:-1]
        for time in StayTime:
            Track.run_Model(np.random.choice(self.MovementModels), time)

        TrackData = Track.get_real_data_all().to_numpy()           # 为了方便增量更新 TrackData是按照行进行时间堆叠即[Time,Columns]
        if self.Flag_withTime:
            TensorTrack = torch.tensor(TrackData.T).to(device)
        else:
            TensorTrack = torch.tensor(TrackData[:,:-1].T).to(device)  # 默认最后一列是timestep

        self.pure_track = TensorTrack.clone().detach()
        self.noisy_track = TensorTrack                                  # 不加噪声就没有噪声。
        return TensorTrack

    def add_noise(self,*args,**kwargs):
        return self.noisy_track

    def get_pure_track(self):
        return self.pure_track.clone().detach()

    def get_noisy_track(self):
        return self.noisy_track.clone().detach()

    def __getitem__(self, idx):
        sample = self.noisy_track[:,idx:idx + self.xWin]
        label = self.noisy_track[:,idx + self.xWin: idx + self.xWin + self.yWin]
        return sample, label

    def __len__(self):
        return self.Simulate_frame - self.xWin - self.yWin + 1

class SNRNoise_Track_Dataset_Generate(Basic_Track_Dataset_Generate):
    def __init__(self,Simulate_frame, dt=0.1, Sigma=0.01,
                 xWin=5, yWin=1, WithTime=False, transpose=True, seed=None):
        super().__init__(Simulate_frame,dt,Sigma,xWin,yWin,seed=seed,Flag_withTime=WithTime)

    def add_noise(self, snr=0):

        def dim_noise(input: torch.Tensor, dim: int, snr=0) -> torch.Tensor:
            def db_to_linear(db_value):
                linear_value = 10 ** (db_value / 20)
                return linear_value

            std = torch.std(input, dim=dim, keepdim=True)
            noise = torch.randn_like(input) * std * db_to_linear(snr)
            return noise

        dataset = copy.copy(self)
        dataset.TensorTrack = dataset.TensorTrack + dim_noise(dataset.TensorTrack, dim=-2, snr=snr)
        dataset.TrackData_noisy = dataset.TensorTrack
        if self.Flag_withTime:
            dataset.TensorTrack[:, -1] = dataset.TensorTrack[:, -1]
            dataset.TrackData_noisy = dataset.TrackData
        return dataset


class CovarianceNoise_Track_Dataset_Generate(Basic_Track_Dataset_Generate):
    default_Cov = torch.diag(torch.tensor([1e1,1e-2,1e-4]*3))
    def __init__(self,Simulate_frame, dt=0.1, Sigma=0.01,
                 xWin=5, yWin=1, WithTime=False, transpose=True, seed=None):
        super().__init__(Simulate_frame,dt,Sigma,xWin,yWin,seed=seed,Flag_withTime=WithTime)

    def add_noise(self,Cov=None,Mean=None):
        if self.Flag_withTime:
            xDim = len(self.pure_track)-1
        else:
            xDim = len(self.pure_track)
        M = MultivariateNormal(Mean if Mean else torch.zeros(xDim).to(device), Cov if Cov else self.default_Cov)
        noise = M.sample([self.Simulate_frame])
        if self.Flag_withTime:
            noisy_track = self.pure_track + noise
        else:
            noisy_track = self.pure_track + noise
        self.noisy_track = noisy_track.clone().detach()
        return self.noisy_track.clone().detach()




# class Random_Track_Dataset_Generate(torch.utils.data.Dataset):
#     Scale_vector = np.array([3000, 10, 1e-3] * 3)
#     MovementModelMap = {"CV": "CVModel",
#                         "CT": "CTxyModel",
#                         "CA": "CAModel"}
#
#     def __init__(self, Simulate_frame, dt=0.1, Sigma=0.01,
#                  xWin=5, yWin=1, WithTime=False, transpose=True, seed=None):
#         super().__init__()
#         MMF = MovementModelFactory()
#         self.SB = SimulationBox()
#         self.SB.SystemCfgUpdate({"Ts": dt,
#                                  "QSigma": Sigma,
#                                  "SimulationTimeTicks": Simulate_frame})
#         self.simFrame = Simulate_frame
#         self.xWin = xWin
#         self.yWin = yWin
#         self.WithTime = WithTime
#         self.Transpose = transpose
#         CVModel = MMF.create('CVModel')(dt, Sigma)
#         CTModel = MMF.create('CTxyModel')(dt, Sigma, -0.35)
#         CAModel = MMF.create('CAModel')(dt, Sigma)
#         self.MovementModels = [CAModel, CTModel, CVModel]
#         self.Track = None
#         self.TrackData = None
#         self.seed = seed
#         self.gen_randomTrack()
#
#     def gen_randomTrack(self, init_point=None, div_num=10):
#         if self.seed:
#             np.random.seed(self.seed)
#         X0 = np.random.rand(9) * self.Scale_vector * np.random.choice([-1, 1], 9) if init_point is None else init_point
#         self.Track = TargetFromKeyframe(self.SB)
#         self.Track.step(X0)
#         ShiftTime = np.r_[0, np.sort(np.random.choice(np.arange(self.simFrame - 1), div_num)), self.simFrame - 1]
#         StayTime = ShiftTime[1:] - ShiftTime[:-1]
#         for time in StayTime:
#             self.Track.run_Model(np.random.choice(self.MovementModels), time)
#
#         self.TrackData = self.Track.get_real_data_all().to_numpy()
#         self.TrackData = torch.tensor(self.TrackData)  # 此时tensor还在cpu
#         self.TrackData_pure = self.TrackData
#         self.TrackData_noisy = None
#         if not self.WithTime:
#             self.TrackData = self.TrackData[:, :9]
#             self.TrackData_pure = self.TrackData
#         return self.Track
#
#     def add_noise(self, snr=0):
#
#         def dim_noise(input: torch.Tensor, dim: int, snr=0) -> torch.Tensor:
#             def db_to_linear(db_value):
#                 linear_value = 10 ** (db_value / 20)
#                 return linear_value
#
#             std = torch.std(input, dim=dim, keepdim=True)
#             noise = torch.randn_like(input) * std * db_to_linear(snr)
#             return noise
#
#         dataset = copy.copy(self)
#         dataset.TrackData = dataset.TrackData + dim_noise(dataset.TrackData, dim=-2, snr=snr)
#         dataset.TrackData_noisy = dataset.TrackData
#         if self.WithTime:
#             dataset.TrackData[:, -1] = dataset.TrackData[:, -1]
#             dataset.TrackData_noisy = dataset.TrackData
#         return dataset
#
#     def get_pure_track(self):  # 使用这个直接获得轨迹tensor
#         return self.TrackData_pure
#
#     def get_noisy_track(self, snr=0):  # 使用这个直接获得含有噪声的轨迹tensor,如果还没加噪声，可以添加
#         if self.TrackData_noisy is not None:
#             self.add_noise(snr=snr)
#         return self.TrackData_noisy
#
#     def __getitem__(self, idx):
#         sample = self.TrackData[idx:idx + self.xWin]
#         label = self.TrackData[idx + self.xWin: idx + self.xWin + self.yWin]
#         if self.Transpose:
#             return sample.T, label.T
#         else:
#             return sample, label
#
#     def __len__(self):
#         return self.simFrame - self.xWin - self.yWin + 1


if __name__ == '__main__':
    RTG = SNRNoise_Track_Dataset_Generate(500)
    for i in range(len(RTG)):
        print(RTG[i])