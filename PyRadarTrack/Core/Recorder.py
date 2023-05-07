#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Recorder.py
# @Time      :2023/1/8 12:23 AM
# @Author    :Kinddle
"""
程序中所有需要重放的数据都通过一个recorder进行管理 其基本信息如下：
1. 维护一个DataFrame用来存储数据 其列含义应当在初始化时定义
2. 用numpy的array作为缓存，其块长度默认为五百，但可以更改
3. 仅当缓存块填满或者Recorder的数据被读取时，缓存块内的数据才会被保存在DataFrame中，因此DataFrame不会直接暴露在外

数据一旦录入，不支持部分修改。但可以通过clear指令清空并获取当前所有数据的数据在外界处理后重新load进入
"""
import pandas as pd
import numpy as np


class Recorder:
    def __init__(self, para_list, Buffer_len=500):
        self._Data = pd.DataFrame(columns=para_list)
        self._Buffer_row_num = Buffer_len
        self._Buffer_column_num = len(para_list)
        self._Buffer = None
        self._p = -1
        self._Buffer_init()

    def _Buffer_init(self):
        self._Buffer = np.empty([self._Buffer_row_num, self._Buffer_column_num])
        self._p = -1

    def step(self, X):
        self._p += 1
        if self._p >= self._Buffer_row_num:
            self._DataUpdate()
            self._p += 1
            self._Buffer[self._p, :] = X
        else:
            self._Buffer[self._p, :] = X
        return self

    def get_data(self, index=-1):
        self._DataUpdate()
        if index > len(self._Data):
            return None
        else:
            return self._Data.iloc[index, :]

    def get_data_all(self):
        self._DataUpdate()
        return self._Data.copy(deep=True)

    def _DataUpdate(self):
        save = pd.DataFrame(self._Buffer[:self._p + 1, :],columns=self._Data.columns)
        self._Data = self._Data._append(save, ignore_index=True)
        self._Buffer_init()
        return self._Data

    def clear(self):
        self._DataUpdate()
        rtn = self._Data.copy()
        self._Data = pd.DataFrame(columns=self._Data.columns)
        return rtn

    def load(self, Data: [pd.DataFrame , np.array]):
        self._DataUpdate()
        self._Data.append(Data, ignore_index=True)
        return self

    def setBufferLen(self, BufferLen):
        # New = Recorder(self._Data.columns,BufferLen)
        # New.load(self.clear())
        self._Buffer_row_num = BufferLen
        self._DataUpdate()
        # return New

    def __call__(self, *args, **kwargs):
        return self.step(*args,**kwargs)