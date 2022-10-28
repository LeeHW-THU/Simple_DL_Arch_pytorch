import torch
import torch.nn.functional as F
import tqdm
import numpy
from torch import nn

def train(model, epoch,epoch_num, train_loader):
    """训练"""
    device = torch.device('cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    # 训练模式
    model.train()
    model=model.to(device)

    # 迭代
    for batch_idx, (data, target) in train_loader:
        # 加速

        data, target = data.to(device), target.to(device)

        # 梯度清零
        optimizer.zero_grad()

        output = model(data)

        # 计算损失
        loss = F.nll_loss(output, target).to(device)

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()


        # 打印损失

        train_loader.set_description(f'Epoch [{epoch+1}/{epoch_num}]')
        train_loader.set_postfix(loss='%.8f'%loss.item())