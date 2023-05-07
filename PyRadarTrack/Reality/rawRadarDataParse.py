#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :rawRadarDataParse.py
# @Time      :2023/3/1 1:23 PM
# @Author    :Kinddle
"""
离线处理雷达数据
"""

from ..Core import *


class rawRadarData(BasicObject):
    pass


class rawRadarDataParse(BasicObject):
    def __init__(self):
        super(rawRadarDataParse, self).__init__()
        self.Data = rawRadarData()




if __name__ == '__main__':  
    cj=1j
    C=3e8
    
    ## 这些变量你后面处理信号可能要用
    frame_num=1
    sample_len=512
    pulse_num=16
    TX=12
    RX=16
    frame_len=TX*RX*pulse_num*sample_len
    total_len=TX*RX*pulse_num*sample_len*frame_num
    
    # 500M
    S=1.4286e13
    fs=1.4603e7
    deta_range=fs*C/sample_len/S/2
    
    ## 读信号，只需要改frame_index 和 file_name
    frame_index=1# 读第几帧
    file_name=['CV0421_real_xuanting.dat'] # 改这里，改文件路径
    class ClsA:
        pass
    load_adc_param = ClsA()
    load_adc_param.file_name=file_name
    load_adc_param.frame_index=frame_index
    load_adc_param.numSamplePerChirp=sample_len
    load_adc_param.numChirpPerLoop=12
    load_adc_param.numLoops=pulse_num
    load_adc_param.numRXPerDevice=4
    load_adc_param.DeviceNums=4

    def LoadAdcData(param):
        # ref
        file_name = param.file_name
        frame_index = param.frame_index
        numSamplePerChirp = param.numSamplePerChirp
        numChirpPerLoop = param.numChirpPerLoop
        numLoops = param.numLoops
        numRXPerDevice = param.numRXPerDevice
        CPBytes = 0 # 16
        DeviceNums = param.DeviceNums
        Expected_Bytes_SamplesPerChirp = (numSamplePerChirp * numRXPerDevice * 2 * 2 + CPBytes) * DeviceNums
        Expected_Nums_SamplesPerChirp = numSamplePerChirp * numRXPerDevice * 2 * DeviceNums
        Expected_Bytes_SamplesPerFrame = Expected_Bytes_SamplesPerChirp * numChirpPerLoop * numLoops # numSamplePerChirp * numChirpPerLoop * numLoops * numRXPerDevice * 2 * 2
        Expected_Num_SamplesPerFrame = numSamplePerChirp * numChirpPerLoop * numLoops * numRXPerDevice * 2 * DeviceNums
        frameIdx = frame_index
        adcData = []

        with open(file_name,"rb") as fp:
            fp.seek((frameIdx - 1) * Expected_Bytes_SamplesPerFrame, 0)
            # adcData1 = fread(fp, inf, 'int16')
            for i in range(numLoops * numChirpPerLoop):
                tmp_data = fp.read(Expected_Nums_SamplesPerChirp*2) #  'int16'
                adcData.append(tmp_data)
                print(i)
            # for i=1:numLoops * numChirpPerLoop
            # fseek(fp, DeviceNums * CPBytes, 'cof')
            # Data = fread(fp, Expected_Nums_SamplesPerChirp, 'int16')
            # adcData = [adcData, Data.']

        # end
        # fclose(fp)
        # neg = logical(bitget(adcData1, 16))
        # adcData1(neg) = adcData1(neg) - 2 ^ 16
        # #



        # adcData1 = adcData(1:2: end) + sqrt(-1) * adcData(2: 2:end)
        # # adcData1Complex = reshape(adcData1, DeviceNums, numRXPerDevice, numSamplePerChirp, numChirpPerLoop * numLoops)
        #
        # adcData2 = zeros(numSamplePerChirp, numLoops, numChirpPerLoop * DeviceNums * numRXPerDevice)
        # for m=1:numLoops
        # for n=1:numChirpPerLoop
        # begin_pos = (((m - 1) * numChirpPerLoop + n - 1) * DeviceNums * numRXPerDevice) * numSamplePerChirp
        # for k=1:DeviceNums * numRXPerDevice
        # row_index = floor((k - 1) / DeviceNums)
        # col_index = mod(k - 1, DeviceNums)
        # adcTmp = adcData1(
        #     begin_pos + k:DeviceNums * numRXPerDevice: begin_pos + DeviceNums * numRXPerDevice * numSamplePerChirp)
        # adcTmp(1) = 0
        # channel_index = col_index * numRXPerDevice + mod(row_index, numRXPerDevice) + 1
        # adcData2(:, m, (n - 1) * DeviceNums * numRXPerDevice + channel_index)=adcTmp
        # end
        # end
        # end
        #
        # adc_data = adcData2
        # end
        return adcData
        # return adc_data


    adc_data_t = LoadAdcData(load_adc_param)






