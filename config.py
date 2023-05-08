#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :config.py
# @Time      :2023/5/4 3:22 PM
# @Author    :Oliver
import os
# import sys
import datetime
import torch
WORK_ROOT = os.getcwd()
OUTPUT_ROOT_DIR = os.path.join(WORK_ROOT,"output/")
LOG_ROOT_DIR = os.path.join(WORK_ROOT,"logs/")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Path_OutputDir(filename,leaf_dir=""):
    dir = os.path.join(OUTPUT_ROOT_DIR, leaf_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, filename)
def Path_LogDir(filename,leaf_dir=""):
    dir = os.path.join(LOG_ROOT_DIR, leaf_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, filename)


def getStrTime(Date=True, Time=True):
    format_str=""
    if Date:
        format_str +="%Y-%m-%d"
    if Date and Time:
        format_str += "__"
    if Time:
        format_str += "%H-%M-%S"
    return datetime.datetime.now().strftime(format_str)
