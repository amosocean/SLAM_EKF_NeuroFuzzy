#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Trainer.py
# @Time      :2023/5/4 6:53 PM
# @Author    :Oliver
import logging

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import gif,os,json
from config import *
from FuzzyModel.MyModel import *

class BasicTrainer(object):
    device = device

    def __init__(self, model, loader_train, loader_test, optimizer, lrScheduler,logName=None, lossFunc=None):
        self.train_loss = None
        self.test_loss = None
        if logName is None:
            logName = "default_logger"
            self.log = logging.getLogger(logName)
            self.log.setLevel("INFO")
            self.log.info("Trainer ready: logged by default_logger.")
        else:
            self.log = logging.getLogger(logName)
            self.log.info(f"Trainer ready: logged by {logName}.")
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.optimizer = optimizer
        self.lrScheduler = lrScheduler
        # self.lossFunc = lossFunc if lossFunc is not None else lambda pred,gts:torch.sqrt(F.mse_loss(pred, gts))
        self.reset()
        self.tmp_save_data_pred = []
        self.tmp_save_data_real = []

    def work(self, ifRecord=True, ifEval=False):
        if ifEval:
            self.model.eval()
            loader = self.loader_test
        else:
            self.model.train()
            loader = self.loader_train
        self.tmp_save_data_pred.clear()
        self.tmp_save_data_real.clear()

        loss = self.workStep(loader, ifRecord, ifEval)

        self.lrScheduler.step()
        if ifRecord:
            if ifEval:
                self.save_data_test.update({self.epoch_count: [self.tmp_save_data_pred, self.tmp_save_data_real]})
                self.save_data_lr.update({self.epoch_count: self.optimizer.param_groups[0]['lr']})
            else:
                self.save_data_train.update({self.epoch_count: [self.tmp_save_data_pred, self.tmp_save_data_real]})
        return loss

    def workStep(self, loader, ifRecord=True, ifEval=False):
        total_loss = 0
        for batch in loader:
            sample = batch[0].to(self.device)
            gts = batch[1].to(self.device)   #转移到gpu
            batch_len = sample.shape[0]
            pred = self.model(sample)
            loss = F.mse_loss(pred, gts)
            if not ifEval:
                self.optimizer.zero_grad()  # pytorch会积累梯度，在优化每个batch的权重的梯度之前将之前计算出的每个权重的梯度置0
                loss.backward()  # 在最后一个张量上调用反向传播方法，在计算图中计算权重的梯度
                self.optimizer.step()  # 使用预先设置的学习率等参数根据当前梯度对权重进行更新
            total_loss += loss * batch_len
            if ifRecord:
                self.tmp_save_data_pred.extend(pred.tolist())
                self.tmp_save_data_real.extend(gts.tolist())
        loss = total_loss / len(loader.dataset)

        return loss

    def train(self, ifRecord=True):
        return self.work(ifRecord, False)

    def eval(self, ifRecord=True):
        return self.work(ifRecord, True)

    def reset(self):
        self.epoch_count = 0
        self.save_data_train = {}
        self.save_data_test = {}
        self.save_data_lr = {}

    def run(self, epoch_num, div=2,show_loss=False):
        train_losses = {}
        test_losses = {}
        for epoch in range(epoch_num):
            train_loss = self.train()
            train_loss=train_loss.detach().cpu().numpy()
            train_losses.update({epoch: train_loss})
            show_str = f"epoch:{epoch + 1},train_loss:{train_loss}"
            if (epoch + 1) % div == 0:
                test_loss = self.eval()
                test_loss=test_loss.detach().cpu().numpy()
                test_losses.update({epoch: test_loss})
                show_str = f"epoch:{epoch + 1},train_loss:{train_loss},test_loss:{test_loss}"
            print("\r"+show_str, end="")
            self.log.info(show_str)
            self.epoch_count += 1
        self.train_loss = train_losses
        self.test_loss = test_losses
        if show_loss:
            self.show()
        return train_losses, test_losses

    def show(self):
        fig = self.drawLossFig()
        fig.show()

    def drawLossFig(self):
        train_loss, test_loss = self.train_loss,self.test_loss
        fig = plt.figure()
        train_x = np.array([i for i, j in train_loss.items()])
        train_y = np.array([j for i, j in train_loss.items()])
        test_x = np.array([i for i, j in test_loss.items()])
        test_y = np.array([j for i, j in test_loss.items()])
        plt.plot(train_x, train_y, label="Train")
        plt.plot(test_x, test_y, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        return fig

    def MFAnalyze(self,File_Path):
        Ant_F = self.model.Inference.Ant_Function
        Height = self.model.Defuzzifier.para_height.detach()
        sample = torch.linspace(0, 1, 100)[:, None, None]
        draw_data = Ant_F(sample).detach()

        @gif.frame
        def plot_(rule):
            ant = draw_data[:, :, rule]
            height = Height[:, :, rule]
            # real, pred = draw_data_pred_x[epoch]
            fig = plt.figure(figsize=[10, 5])

            plt.subplot(1, 2, 1)
            plt.title(f"Rule-{rule + 1}")
            plt.plot(sample.squeeze(), ant)
            plt.xlim(-0.05, 1.05)
            plt.ylim(0, 1.5)
            plt.legend(["x{}_Ant.".format(i) for i in range(self.model.xDim)], loc="upper right")

            plt.subplot(1, 2, 2)
            plt.vlines(height, 0, 1, label="Con.")
            plt.xlim(min(torch.min(Height), 0) - 0.05, torch.max(Height) + 0.05)
            plt.ylim(0, 1.25)
            plt.legend()

        frame = []
        for k in range(self.model.rule_num):
            of = plot_(k)
            frame.append(of)
        frame[0].save(File_Path, save_all=True, loop=True, append_images=frame[1:],
                      duration=750, disposal=2)

    def ModelSave(self,SaveDir,ModelName="model"):
        SaveDir = Path_OutputModelDir("",SaveDir)
        Model_save_path = os.path.join(SaveDir, f"{ModelName}.pt")
        Json_save_path = os.path.join(SaveDir,f"{ModelName}_config.json")
        torch.save(self.model.state_dict(), Model_save_path)

        with open(Json_save_path,"w") as F:
            json.dump({"model_class": self.model.__class__.__name__,
                       "state_dict_path": Model_save_path,
                       "para":self.model.get_init_para()},F)
        return Json_save_path
    @staticmethod
    def LoadModelByJson(json_path):
        with open(json_path,'r') as F:
            data = json.load(F)
        ModelClassName,Parameters_path,InitPara=data["model_class"],data["state_dict_path"],data["para"]
        model = eval(f"{ModelClassName}(**{InitPara})")
        model.to(device)
        model.load_state_dict(torch.load(Parameters_path))
        return model


class MSETrainer(BasicTrainer):
    def __init__(self, model, loader_train, loader_test, optimizer, lrScheduler,logName=None):
        super().__init__(model, loader_train, loader_test, optimizer, lrScheduler,logName=logName)


class RMSETrainer(BasicTrainer):
    def __init__(self, model, loader_train, loader_test, optimizer, lrScheduler,logName=None):
        super().__init__(model, loader_train, loader_test, optimizer, lrScheduler,logName=logName)

    def workStep(self, loader, ifRecord=True, ifEval=False):
        total_loss = 0
        for batch in loader:
            sample = batch[0].to(self.device)
            gts = batch[1].to(self.device)
            batch_len = sample.shape[0]
            pred = self.model(sample).squeeze(-1)
            loss = torch.sqrt(F.mse_loss(pred, gts))
            if not ifEval:
                self.optimizer.zero_grad()  # pytorch会积累梯度，在优化每个batch的权重的梯度之前将之前计算出的每个权重的梯度置0
                loss.backward()  # 在最后一个张量上调用反向传播方法，在计算图中计算权重的梯度
                self.optimizer.step()  # 使用预先设置的学习率等参数根据当前梯度对权重进行更新
            total_loss += torch.square(loss) * batch_len
            if ifRecord:
                self.tmp_save_data_pred.extend(pred.tolist())
                self.tmp_save_data_real.extend(gts.tolist())
        loss = torch.sqrt(total_loss / len(loader.dataset))
        return loss
