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
from PyRadarTrack.Model.FilterModel import IMMFilterModel,BasicEKFModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
Scale_vector=np.array([3000,10,1e-3]*3)
class Random_Track_Dataset_Generate(torch.utils.data.Dataset):
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
        self.TrackData=torch.tensor(self.TrackData)  #此时tensor还在cpu
        self.TrackData_pure=self.TrackData
        self.TrackData_noisy=None
        if not self.WithTime:
            self.TrackData = self.TrackData[:,:9]
            self.TrackData_pure= self.TrackData
        return self.Track
    
    def add_noise(self,snr=0):
        
        def dim_noise(input:torch.Tensor,dim:int,snr=0)->torch.Tensor:
            
            def db_to_linear(db_value):
                linear_value = 10**(db_value/20)
                return linear_value
            
            std=torch.std(input,dim=dim,keepdim=True)
            noise=torch.randn_like(input)*std*db_to_linear(snr)
            return noise
        dataset=copy.copy(self)
        dataset.TrackData=dataset.TrackData+dim_noise(dataset.TrackData,dim=-2,snr=snr)
        dataset.TrackData_noisy=dataset.TrackData
        if self.WithTime:
            dataset.TrackData[:,-1]=dataset.TrackData[:,-1]
            dataset.TrackData_noisy=dataset.TrackData
        return dataset
    
    def get_pure_track(self): #使用这个直接获得轨迹tensor
        return self.TrackData_pure
    
    def get_noisy_track(self,snr=0):#使用这个直接获得含有噪声的轨迹tensor,如果还没加噪声，可以添加
        if self.TrackData_noisy is not None:
            self.add_noise(snr=snr)
        return self.TrackData_noisy
    
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
    RTG = Random_Track_Dataset_Generate(500)
    for i in range(len(RTG)):
        print(RTG[i])
