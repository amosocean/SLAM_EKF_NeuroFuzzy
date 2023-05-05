#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Trainer.py
# @Time      :2023/5/4 6:53 PM
# @Author    :Oliver
import torch
import torch.nn.functional as F


class BasicTrainer(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, model, loader_train, loader_test, optimizer, lrScheduler, lossFunc=None):
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

        loss = self.workStep(loader,ifRecord,ifEval)

        self.lrScheduler.step()
        if ifRecord:
            if ifEval:
                self.save_data_test.update({self.epoch_count:[self.tmp_save_data_pred,self.tmp_save_data_real]})
                self.save_data_lr.update({self.epoch_count:self.optimizer.param_groups[0]['lr']})
            else:
                self.save_data_train.update({self.epoch_count:[self.tmp_save_data_pred,self.tmp_save_data_real]})
        return loss

    def workStep(self, loader, ifRecord=True,ifEval=False):
        total_loss = 0
        for batch in loader:
            sample = batch[0].to(self.device)
            gts = batch[1].to(self.device)
            batch_len = sample.shape[0]
            pred = self.model(sample).squeeze(-1)
            loss = F.mse_loss(pred, gts)
            if not ifEval:
                self.optimizer.zero_grad()  # pytorch会积累梯度，在优化每个batch的权重的梯度之前将之前计算出的每个权重的梯度置0
                loss.backward()             # 在最后一个张量上调用反向传播方法，在计算图中计算权重的梯度
                self.optimizer.step()       # 使用预先设置的学习率等参数根据当前梯度对权重进行更新
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
    def run(self, epoch_num, div=2):
        train_losses = {}
        test_losses = {}
        for epoch in range(epoch_num):
            train_loss = self.train()
            print(f"\repoch:{epoch+1},train_loss:{train_loss}", end="")
            train_losses.update({epoch:train_loss})
            if (epoch+1) % div == 0:
                test_loss = self.eval()
                print(f"\repoch:{epoch+1},train_loss:{train_loss},test_loss:{test_loss}", end="")
                test_losses.update({epoch: test_loss})
            self.epoch_count += 1
        return train_losses,test_losses

class MSETrainer(BasicTrainer):
    def __init__(self,model, loader_train, loader_test, optimizer, lrScheduler):
        super().__init__(model, loader_train, loader_test, optimizer, lrScheduler)


class RMSETrainer(BasicTrainer):
    def __init__(self,model, loader_train, loader_test, optimizer, lrScheduler):
        super().__init__(model, loader_train, loader_test, optimizer, lrScheduler)

    def workStep(self, loader, ifRecord=True,ifEval=False):
        total_loss = 0
        for batch in loader:
            sample = batch[0].to(self.device)
            gts = batch[1].to(self.device)
            batch_len = sample.shape[0]
            pred = self.model(sample).squeeze(-1)
            loss =torch.sqrt(F.mse_loss(pred, gts))
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

