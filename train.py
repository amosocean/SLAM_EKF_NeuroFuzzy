import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from FuzzyModel.model import FuzzyLayer,FuzzyLayer2
from FuzzyModel.dataset import MyDataset
from FuzzyModel.MyModel import FLSLayer,TSFLSLayer,TrapFLSLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 100
input_dim = 4
batch_size = 5
learning_rate = 10e-5
# rules_num = 16
train_dataset=MyDataset(tao=38, start_index=1001,end_index=1500)
train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=False)

test_dataset=MyDataset(tao=38, start_index=1501,end_index=1995)
test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False)


# customize your own model here:
scale = max(max(test_dataset.series),max(train_dataset.series))
# model = FuzzyLayer(4,16,x_scale=1/scale,y_scale=1/scale).to(device)
model = FLSLayer(input_dim,16).to(device)
model = TrapFLSLayer(input_dim,16).to(device)
# model = TSFLSLayer(input_dim,16).to(device)
model.set_xy_offset_scale(x_scale=1/scale,y_scale=1/scale)



# #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,50], gamma=0.5)


draw_data_train = []
draw_data_test = []
draw_data_Ls = []
draw_data_pred_x = {}
for epoch in range(epoch_num):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        sample, gts = batch[0].to(device), batch[1].to(device)
        preds = model(sample).squeeze()
        loss = torch.sqrt(F.mse_loss(preds, gts))
        optimizer.zero_grad()    # pytorch会积累梯度，在优化每个batch的权重的梯度之前将之前计算出的每个权重的梯度置0
        loss.backward()          # 在最后一个张量上调用反向传播方法，在计算图中计算权重的梯度 
        optimizer.step()         # 使用预先设置的学习率等参数根据当前梯度对权重进行更新
        epoch_loss += loss * train_loader.batch_size
    scheduler.step()             # 步长更新
    loss = epoch_loss / len(train_loader.dataset)
    print("epoch loss:%f, Lr:%f" % (loss,optimizer.param_groups[0]['lr']))
    draw_data_train.append([epoch,float(loss)])
    draw_data_Ls.append([epoch,optimizer.param_groups[0]['lr']])
    # .......
    # 每隔几个epoch在测试集上跑一下
    if epoch % 1 == 0:
        tmp_data_draw_real_x = []
        tmp_data_draw_pred_x = []
        model.eval()
        test_epoch_loss = 0
        for test_batch in test_loader:
            test_sample, test_gts = test_batch[0].to(device), test_batch[1].to(device)
            test_preds = model(test_sample).squeeze()
            tmp_data_draw_pred_x.append(test_preds.detach())
            tmp_data_draw_real_x.append(test_gts)
            loss = torch.sqrt(F.mse_loss(test_preds, test_gts))

            test_epoch_loss += loss * test_loader.batch_size
        test_loss = test_epoch_loss / (len(test_loader.dataset))
        draw_data_pred_x.update({epoch:[tmp_data_draw_real_x,tmp_data_draw_pred_x]})
        print("test loss:%f" % (test_loss))
        draw_data_test.append([epoch,float(loss)])
    # .......
    # 根据条件对指定epoch的模型进行保存 将模型序列化到磁盘的pickle包
    # if 精度最高:
    #     torch.save(model.stat_dict(), f'{model_path}_{time_index}.pth')

if True:
    from matplotlib import pyplot as plt
    plt.figure()
    draw_data_train = np.array(draw_data_train)
    draw_data_test = np.array(draw_data_test)
    # draw_data_Ls = np.array(draw_data_Ls)
    plt.plot(draw_data_train[:,0]+1,draw_data_train[:,1],label="Train")
    plt.plot(draw_data_test[:,0]+1,draw_data_test[:,1],label="Test")
    # plt.plot(draw_data_Ls[:,0]+1,draw_data_Ls[:,1],label="Ls")
    plt.xlabel("Epoch")
    plt.ylabel("Loss(RMSE)")
    plt.legend()
    plt.show()

    import gif
    @gif.frame
    def plot_(epoch):
        real, pred = draw_data_pred_x[epoch]
        fig = plt.figure()

        plt.plot(real,label="real")
        plt.plot(pred,label="pred")
        plt.ylim(0,scale)
        plt.legend()

    frame = []
    for k in draw_data_pred_x.keys():
        of = plot_(k)
        frame.append(of)
    frame[0].save("output/Fuzzy_pred_real.gif",save_all=True, loop=True, append_images=frame[1:],
               duration=10, disposal=2)