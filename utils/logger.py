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
        log_filename = Path_LogDir("{}_LOG_{}.log".format(logName, getStrTime()), getStrTime(True, False))
        self.add_file_Handler(log_filename)

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


    # def __getattr__(self, item):
    #     if item in dir(self.log):
    #         return self.log.__getattr__(item)
    #     else:
    #         super().__getattr__(item)


# class Logger(object):
#     level_relations = {
#         'debug':logging.DEBUG,
#         'info':logging.INFO,
#         'warning':logging.WARNING,
#         'error':logging.ERROR,
#         'crit':logging.CRITICAL
#     }#日志级别关系映射
#
#     def __init__(self,filename,level='info',when='midnight',backCount=3,
#                  fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
#         self.logger = logging.getLogger(filename)
#         format_str = logging.Formatter(fmt)#设置日志格式
#         self.logger.setLevel(self.level_relations.get(level))#设置日志级别
#         sh = logging.StreamHandler()#往屏幕上输出
#         sh.setFormatter(format_str) #设置屏幕上显示的格式
#         th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
#         th.setFormatter(format_str)#设置文件里写入的格式
#         self.logger.addHandler(sh) #把对象加到logger里
#         self.logger.addHandler(th)

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





