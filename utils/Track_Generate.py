#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Track_Generate.py
# @Time      :2023/5/7 6:52 PM
# @Author    :Oliver
import torch
import numpy as np
from PyRadarTrack.Model import *
from PyRadarTrack.Simulate import *
from PyRadarTrack.Model.FilterModel import IMMFilterModel,BasicEKFModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
Scale_vector=np.array([3000,10,1e-3]*3)
class Random_Track_Generate(torch.utils.data.Dataset):
    def __init__(self,Simulate_frame,dt=0.1,Sigma=0.01,
                 xWin=5,yWin=1,WithTime=False,transpose=True,seed=None):
        super().__init__()
        MMF = MovementModelFactory()
        self.SB = SimulationBox()
        self.SB.SystemCfgUpdate({"Ts": dt,
                            "QSigma": Sigma,
                            "SimulationTimeTicks": Simulate_frame})
        self.simFrame = Simulate_frame
        self.xWin = xWin
        self.yWin = yWin
        self.WithTime = WithTime
        self.Transpose = transpose
        CVModel = MMF.create('CVModel')(dt, Sigma)
        CTModel = MMF.create('CTxyModel')(dt, Sigma, -0.35)
        CAModel = MMF.create('CAModel')(dt, Sigma)
        self.MovementModels = [CAModel,CTModel,CVModel]
        self.Track = None
        self.TrackData = None
        self.seed=seed
        self.gen_randomTrack()

    def gen_randomTrack(self,init_point=None,div_num=10):
        if self.seed:
            np.random.seed(self.seed)
        X0 = np.random.rand(9)*Scale_vector * np.random.choice([-1,1],9) if init_point is None else init_point
        self.Track = TargetFromKeyframe(self.SB)
        self.Track.step(X0)
        ShiftTime = np.r_[0, np.sort(np.random.choice(np.arange(self.simFrame-1),div_num)),self.simFrame-1]
        StayTime = ShiftTime[1:]-ShiftTime[:-1]
        for time in StayTime:
            self.Track.run_Model(np.random.choice(self.MovementModels),time)

        self.TrackData = self.Track.get_real_data_all().to_numpy()
        self.TrackData=torch.tensor(self.TrackData).to(device)
        if not self.WithTime:
            self.TrackData = self.TrackData[:,:9]
        return self.Track
    def __getitem__(self, idx):
        sample = self.TrackData[idx:idx+self.xWin]
        label = self.TrackData[idx + self.xWin: idx + self.xWin+self.yWin]
        if self.Transpose:
            return sample.T,label.T
        else:
            return sample,label

    def __len__(self):
        return self.simFrame - self.xWin - self.yWin +1

if __name__ == '__main__':
    RTG = Random_Track_Generate(500)
    for i in range(len(RTG)):
        print(RTG[i])
