import torch
import numpy as np
from torch.utils.data import DataLoader

# import matplotlib.pyplot as plt
# import csv
"""
Mackey-Glass time series
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tao=17, start_index=None, end_index=None):
        # =============================================================================
        # Define hyperparameters
        # =============================================================================
        assert start_index and end_index, "Need start_index and end_index"
        assert end_index > start_index, "Start_index should less than end_index"
        self.window_size = 4
        self.start_index = start_index
        self.end_index = end_index
        t_min = tao + 1
        t_max = end_index + self.window_size + 1
        beta = 0.2
        gamma = 0.1
        n = 10
        x = []
        for i in range(1, t_min):
            x.append(0.0)
        x.append(1.2)

        for t in range(t_min, t_max):
            h = x[t - 1] + (beta * x[t - tao - 1] / (1 + np.power(x[t - tao - 1], n))) - (gamma * x[t - 1])
            # h = float("{:0.4f}".format(h))
            x.append(h)
        self.series = torch.Tensor(x).to(device)
        # print(self.series.size())

    def __getitem__(self, idx):
        idx = idx + self.start_index
        sample = self.series[idx:idx + self.window_size]
        label = self.series[idx + self.window_size]
        return sample, label

    def __len__(self):
        return self.end_index - self.start_index + 1


if __name__ == "__main__":
    train_dataset = MyDataset(start_index=1001, end_index=1504)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=False)
    i = 1
    print(len(train_loader))
    for batch in train_loader:
        print(batch[0])
        # print(ground)
        i = i + 1
        print(i)
