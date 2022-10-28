import torch
from torch import nn


def get_loss(config):
    name = config['loss']['name']
    loss = None
    if name == 'nll_loss':
        loss = nn.NLLLoss(reduction='sum')
    if name == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    return loss
