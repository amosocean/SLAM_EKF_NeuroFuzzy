#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :read_and_plot.py
# @Time      :2023/4/14 8:39 PM
# @Author    :Kinddle
from tools import *


def read_world(filename):
    rtn = pd.DataFrame(columns=["id", "x", "y"])
    # rtn_dt = np.dtype({'names': ['id','x','y'],
    #                     'formats': [int, float,float]})

    with open(filename, "r") as F:
        line = " "
        while line:
            line = F.readline()
            if line:
                id, x, y = line.split(None)

                rtn = rtn.append({"id": int(id)-1,
                                  "x": float(x),
                                  "y": float(y)}, ignore_index=True)
    return rtn


def read_data(filename):
    rtn = []
    basic_dict = {"Odometry": None, "Sensor": []}
    with open(filename, "r") as F:
        line = " "
        while line:
            line = F.readline()
            if line:
                Type, p1, p2, p3 = line.split(None)
                if Type == "ODOMETRY":
                    rtn.append(basic_dict.copy())
                    basic_dict = {"Odometry": None, "Sensor": []}
                    basic_dict["Odometry"] = {"r1":eval(p1),"t":eval(p2),"r2":eval(p3)}
                elif Type == "SENSOR":
                    basic_dict["Sensor"].append({"id":eval(p1)-1,"range":eval(p2),"bearing":eval(p3)})
                else:
                    raise IOError("未知的数据类型")
    return rtn[1:]