#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :logger.py
# @Time      :2023/5/6 2:01 PM
# @Author    :Oliver
import sys

import logging
from logging import handlers
from config import *

file_log_format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
console_log_format = '%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s: %(message)s'


class rootlogger(object):
    def __init__(self, logName):
        self.log = logging.getLogger(logName)
        self.log.setLevel("DEBUG")
        self.add_stream_Handler()
        self.log_filename = Path_LogDir("{}_LOG_{}.log".format(logName, getStrTime()), getStrTime(True, False))
        self.add_file_Handler(self.log_filename)

    def silence(self):
        self.log.setLevel(100)
    def add_stream_Handler(self,stream=sys.stderr, level="WARNING"):
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        formatter = logging.Formatter(console_log_format)
        handler.setFormatter(formatter)
        self.log.addHandler(handler)


    def add_file_Handler(self,filename,level="DEBUG"):
        handler = logging.FileHandler(filename=filename)
        handler.setLevel(level)
        formatter = logging.Formatter(file_log_format)
        handler.setFormatter(formatter)
        self.log.addHandler(handler)


class MarkdownEditor(object):
    def __init__(self):
        self.log_path=None
        self.save_path = None
        self.source_dir = None
        # self.data = ""
        self.data = "## This is a detail log..\n"

    def init_By_logger(self,logger:rootlogger):
        NewDir = logger.log_filename[:-4]
        self.save_path = os.path.join(NewDir,"log.md")
        self.source_dir = os.path.join(NewDir,"source")
        if not os.path.exists(NewDir):
            os.makedirs(NewDir)
            os.makedirs(self.source_dir)
        return self

    def add_line(self,Str:str):
        self.data += Str + "\n"
        return self
    def add_figure(self, Fig_filename, figData=None, cpFigPath=None,
                   altText="default", describe=""):
        Fig_Save_Path = os.path.join(self.source_dir,Fig_filename)
        if cpFigPath is None:
            self._saveFig(figData,Fig_Save_Path)
        else:
            self._cpFig(cpFigPath,Fig_Save_Path)

        gram = f"![{altText}]({Fig_Save_Path})"
        self.add_line(gram)
        self.add_line(describe)
        return self

    def _saveFig(self,FigData,FigPath):
        FigData.savefig(FigPath)

    def _cpFig(self,source_path,determine_Path):
        with open(source_path, 'rb') as rStream:
            container = rStream.read()
            with open(determine_Path, 'wb') as wStream:
                wStream.write(container)

    def saveMD(self):
        with open(self.save_path,"w") as F:
            F.write(self.data)
        return self.save_path


if __name__ == '__main__':
    log = rootlogger('test_for_logger')
    log.log.debug('debug')
    log.log.info('info')
    log.log.warning('警告')
    log.log.error('报错')
    log.log.critical('严重')
    # Logger('error.log', level='error').logger.error('error')


# formatter_1 = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
#
#
# # logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
# #                     level=logging.DEBUG)
# if __name__ == '__main__':
#     logging.debug('debug级别，一般用来打印一些调试信息，级别最低')
#     logging.info('info级别，一般用来打印一些正常的操作信息')
#     logging.warning('waring级别，一般用来打印警告信息')
#     logging.error('error级别，一般用来打印一些错误信息')
#     logging.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')





