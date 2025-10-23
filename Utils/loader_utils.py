from torch_geometric.data import InMemoryDataset
from torch.utils.data import DataLoader
import torch
import numpy as np


device = torch.device("cuda:0")
class SynergyEncoderDataset(InMemoryDataset):
    def __init__(self, Combo_data, Y_data, device=None):
        # context:细胞系，Y：LABEL，二分类，这里的maxcompoundlen是相当于batch-size吗？device="cpu"
        self.combo_data = Combo_data
        self.y = Y_data
        self.device = device
        self.len = len(self.y)  # 具体有多少数据

    def __len__(self):
        return self.len



    def __getitem__(self, index):
        combo = self.combo_data[index]
        label = self.y[index]
        return [torch.LongTensor(combo), torch.LongTensor([int(label)])]



def collate(batch):
    # 获取当前batch的样本数量
    batch_size = len(batch)
    # 分别将样本的输入和标签进行组合
    inputs = [item[0] for item in batch]  # 样本的输入
    labels = [item[1] for item in batch]  # 样本的标签

    # 对样本进行预处理，如将文本转换为数值向量等

    # 将inputs和labels转换为模型所需的数据类型和数据结构
    inputs = torch.stack(inputs)  # 将输入列表堆叠为张量
    labels = torch.LongTensor(labels)  # 转换为张量

    return inputs, labels


def define_dataloader(synergy=None, batch_size=None, train=True):
    # 划分训练集
    Combo_data = synergy[:, 0:-1]
    Y_data = synergy[:, 3]
    # 划分测试集
    train_dataset = SynergyEncoderDataset(Combo_data, Y_data, device=device)
    if train:
        trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    else:
        trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return trainLoader