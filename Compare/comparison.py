#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :comparison.py
# @Time      :2023/5/17 10:47 AM
# @Author    :Oliver

class BasicComparison(object):
    def __init__(self):
        self.modelCount = 0
        self.modelDict = {}
        pass

    def add_model(self, model, Name=None):
        if Name is None:
            Name = str(self.modelCount)
            self.modelCount+=1
        self.modelDict.update({Name: {"model":model}})
        return self

    def add_track(self,track):
        pass




model_paths = {}        # {Name:modelDir}
# region 生成测试轨迹

# endregion

# region 卡尔曼滤波

# endregion

# region FLS滤波

# endregion




if __name__ == '__main__':
    pass








