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
from torch.utils.data import DataLoader
from PyRadarTrack.Model.FilterModel import IMMFilterModel, BasicEKFModel
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
# TRACK_BUFFER_LEN = 500


class Raw_Track(object):
    Scale_vector = np.array([3000, 10, 1e-3] * 3)
    MovementModelNameMap = {"CV": "CVModel",
                            "CT": "CTxyModel",
                            "CA": "CAModel"}

    def __init__(self, dt=0.1, Sigma=0.01, UsedModel=None):
        # self.seed = seed
        # self.Simulate_frame = Simulate_frame
        # region --准备运动模型
        self.SB = SimulationBox()
        self.SB.SystemCfgUpdate({"Ts": dt,
                                 "QSigma": Sigma,
                                 # "SimulationTimeTicks": TRACK_BUFFER_LEN,
                                 })
        self.MMF = MovementModelFactory()
        self.MovementModels = []
        if UsedModel is None:
            UsedModel = self.MovementModelNameMap.keys()
        if "CV" in UsedModel:
            self.MovementModels.append(self.MMF.create('CVModel')(dt, Sigma))
        if "CT" in UsedModel:
            self.MovementModels.append(self.MMF.create('CTxyModel')(dt, Sigma, -0.35))
        if "CA" in UsedModel:
            self.MovementModels.append(self.MMF.create('CAModel')(dt, Sigma))
        # endregion

        # region-- 初始化中间变量
        # 用于存储最近一次生成的轨迹及其模型标签
        self.pure_track = None
        self.ModelLabel = []
        # endregion

        # self.generate_randomTrack()

    def generate_randomTrack(self, Simulate_frame, init_point=None, div_num=10, Flag_withTime=False,seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.ModelLabel.clear()
        X0 = np.random.randn(9) * self.Scale_vector if init_point is None else init_point
        Track = TargetFromKeyframe(self.SB)
        Track.step(X0)
        ShiftTime = np.r_[
            0, np.sort(np.random.choice(np.arange(Simulate_frame - 1), div_num)), Simulate_frame - 1]
        StayTime = ShiftTime[1:] - ShiftTime[:-1]
        for time in StayTime:
            ModelID = np.random.choice(np.arange(len(self.MovementModels)))
            Track.run_Model(self.MovementModels[ModelID], time)
            self.ModelLabel.extend([ModelID] * time)

        TrackData = Track.get_real_data_all().to_numpy()  # 为了方便增量更新 TrackData是按照行进行时间堆叠即[Time,Columns]
        if Flag_withTime:
            TensorTrack = torch.tensor(TrackData.T, dtype=torch.float32)
        else:
            TensorTrack = torch.tensor(TrackData[:, :-1].T, dtype=torch.float32)  # 默认最后一列是timestep

        self.pure_track = TensorTrack.clone().detach()
        # self.Measure = TensorTrack  # 不加噪声就没有噪声。
        return TensorTrack

    def get_pure_track(self):
        return self.pure_track.clone().detach()

    def get_model_label(self):
        return torch.tensor(self.ModelLabel)


class BasicTrackDataset_linerMeasure(torch.utils.data.Dataset, Raw_Track):

    def __init__(self, Simulate_frame, dt=0.1, Sigma=0.01, xWin=5, yWin=5, seed=None,
                 UsedModel=None,
                 Flag_withTime=False):
        super().__init__(dt, Sigma, UsedModel)
        self.xWin = xWin
        self.yWin = yWin
        self.seed = seed
        self.Simulate_frame = Simulate_frame
        self.Flag_withTime = Flag_withTime
        self.Flag_Noisy = False
        # self.Flag_transpose = Flag_transpose

        # # region --准备运动模型
        # self.SB = SimulationBox()
        # self.SB.SystemCfgUpdate({"Ts": dt,
        #                          "QSigma": Sigma,
        #                          "SimulationTimeTicks": Simulate_frame})
        # self.MMF = MovementModelFactory()
        # self.MovementModels = []
        # if UsedModel is None:
        #     UsedModel = self.MovementModelNameMap.keys()
        # if "CV" in UsedModel:
        #     self.MovementModels.append(self.MMF.create('CVModel')(dt, Sigma))
        # if "CT" in UsedModel:
        #     self.MovementModels.append(self.MMF.create('CTxyModel')(dt, Sigma, -0.35))
        # if "CA" in UsedModel:
        #     self.MovementModels.append(self.MMF.create('CAModel')(dt, Sigma))
        #
        # # endregion

        # region-- 初始化中间变量
        # self.pure_track = None
        self.Measure = None
        # self.ModelLabel = []
        # endregion

        self.Measure = self.gen_randomTrack()

    def gen_randomTrack(self,init_point=None,div_num=10):
        return super().generate_randomTrack(Simulate_frame=self.Simulate_frame,
                                            init_point=init_point,div_num=div_num,
                                            Flag_withTime=self.Flag_withTime,
                                            seed=self.seed)
    # def gen_randomTrack(self, init_point=None, div_num=10):
    #     if self.seed:
    #         np.random.seed(self.seed)
    #     self.ModelLabel.clear()
    #     X0 = np.random.randn(9) * self.Scale_vector if init_point is None else init_point
    #     Track = TargetFromKeyframe(self.SB)
    #     Track.step(X0)
    #     ShiftTime = np.r_[
    #         0, np.sort(np.random.choice(np.arange(self.Simulate_frame - 1), div_num)), self.Simulate_frame - 1]
    #     StayTime = ShiftTime[1:] - ShiftTime[:-1]
    #     for time in StayTime:
    #         ModelID = np.random.choice(np.arange(len(self.MovementModels)))
    #         Track.run_Model(self.MovementModels[ModelID], time)
    #         self.ModelLabel.extend([ModelID] * time)
    #
    #     TrackData = Track.get_real_data_all().to_numpy()  # 为了方便增量更新 TrackData是按照行进行时间堆叠即[Time,Columns]
    #     if self.Flag_withTime:
    #         TensorTrack = torch.tensor(TrackData.T, dtype=torch.float32)
    #     else:
    #         TensorTrack = torch.tensor(TrackData[:, :-1].T, dtype=torch.float32)  # 默认最后一列是timestep
    #
    #     self.pure_track = TensorTrack.clone().detach()
    #     self.Measure = TensorTrack  # 不加噪声就没有噪声。
    #     return TensorTrack
    def add_noise(self, *args, **kwargs):
        self.Measure = self.pure_track.detach().clone()
        return self

    def get_measure(self):
        return self.Measure.clone().detach()

    def normalize(self):  # 采用Min-Max归一化
        assert self.Flag_Noisy, "建议归一化前加入噪声"
        indices = torch.tensor([0, 3, 6])  # 第1行、第4行和第7行的索引(x,y,z位移的索引)
        selected_rows = torch.index_select(self.Measure, dim=0, index=indices)
        max_value = torch.max(selected_rows)  # 求x,y,z最大值
        min_value = torch.min(selected_rows)  # 求x,y,z最小值
        self.Measure = (self.Measure - min_value) / (max_value - min_value)
        self.pure_track = (self.pure_track - min_value) / (max_value - min_value)
        return self

    def __getitem__(self, idx):
        sample = self.Measure[:, idx:idx + self.xWin]
        # label = self.Measure[:,idx + self.xWin: idx + self.xWin + self.yWin]
        label = self.pure_track[:, idx + self.xWin: idx + self.xWin + self.yWin]
        return sample, label

    def __len__(self):
        return self.Simulate_frame - self.xWin - self.yWin + 1


class SNRNoise_Track_Dataset_LinerMeasure(BasicTrackDataset_linerMeasure):
    def __init__(self, Simulate_frame, dt=0.1, Sigma=0.01,
                 xWin=5, yWin=1, noise_snr=None, WithTime=False, UsedModel=None, transpose=True, seed=None):
        super().__init__(Simulate_frame, dt, Sigma, xWin, yWin, UsedModel=UsedModel, seed=seed, Flag_withTime=WithTime)

    def add_noise(self, snr=0):
        if not self.Flag_Noisy:
            def dim_noise(input: torch.Tensor, dim: int, snr=0) -> torch.Tensor:
                def db_to_linear(db_value):
                    linear_value = 10 ** (db_value / 20)
                    return linear_value

                std = torch.std(input, dim=dim, keepdim=True)
                noise = torch.randn_like(input) * std * db_to_linear(snr)
                return noise

            TrackData = self.get_pure_track()
            self.Measure = TrackData + dim_noise(TrackData, dim=-1, snr=snr)
            self.Flag_Noisy = True
        return self


class CovarianceNoise_Track_Dataset_LinerMeasure(BasicTrackDataset_linerMeasure):
    default_Cov = torch.diag(torch.tensor([1e1, 1e-2, 1e-4] * 3))

    def __init__(self, Simulate_frame, dt=0.1, Sigma=0.01,
                 xWin=5, yWin=1, UsedModel=None, WithTime=False, transpose=True, seed=None):
        super().__init__(Simulate_frame, dt, Sigma, xWin, yWin, UsedModel=UsedModel, seed=seed, Flag_withTime=WithTime)

    def add_noise(self, Cov=None, Mean=None):
        if self.Flag_withTime:
            xDim = len(self.pure_track) - 1
        else:
            xDim = len(self.pure_track)
        M = MultivariateNormal(Mean if Mean is not None else torch.zeros(xDim).to(device),
                               Cov if Cov is not None else self.default_Cov)
        noise = M.sample(torch.Size([self.Simulate_frame])).T
        if self.Flag_withTime:
            Measure = self.pure_track[:-1] + noise
        else:
            Measure = self.pure_track + noise
        self.Measure = Measure.clone().detach()
        return self


# class SNRNoise_Track_Dataset_Normalized(SNRNoise_Track_Dataset_LinerMeasure):
#     def __init__(self, Simulate_frame, dt=0.1, Sigma=0.01,
#                  xWin=5, yWin=1, WithTime=False, transpose=True, seed=None):
#         super(SNRNoise_Track_Dataset_Normalized, self).__init__(Simulate_frame, dt, Sigma, xWin, yWin, seed=seed,
#                                                                 Flag_withTime=WithTime)


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
#     def get_Measure(self, snr=0):  # 使用这个直接获得含有噪声的轨迹tensor,如果还没加噪声，可以添加
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
    RTG = SNRNoise_Track_Dataset_LinerMeasure(500)
    for i in range(len(RTG)):
        print(RTG[i])
