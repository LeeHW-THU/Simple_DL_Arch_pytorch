import torch
import torch.nn.functional as F
import tqdm
import numpy
from torch import nn
from units import get_optimizer
from loss import *


def train(model, epoch, config, train_loader):
    """Start train"""
    if config['mode']['name'] == 'gpu':
        device = torch.device('cuda:{}'.format(str(config['mode']['ids'])))
    else:
        device = torch.device('cpu')

    # Train mode
    model.train()
    model = model.to(device)

    # Define optimizer
    optimizer = get_optimizer(model.parameters(), config)
    loss_fun = get_loss(config)

    # Start iter
    for batch_idx, (data, target) in train_loader:
        # Speedup
        data, target = data.to(device), target.to(device)

        # Set grad to ZERO
        optimizer.zero_grad()

        # Input Model
        output = model(data)

        # Calculate loss
        loss = loss_fun(output, target).to(device)

        # Backward
        loss.backward()

        # Update grad
        optimizer.step()

        # Print Loss
        epoch_num = config['epoch_max']
        train_loader.set_description(f'Epoch [{epoch+1}/{epoch_num}]')
        train_loader.set_postfix(loss='%.8f' % loss.item())
