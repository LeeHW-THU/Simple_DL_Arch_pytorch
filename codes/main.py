import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import yaml
from units import *
from dataloader import load_data
from model import *
from train import *
from test import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, default='./options.yaml', type=str, help='Select GPUs')
    args = parser.parse_args()

    config = load_conf(args.config)
    set_envs(config)
    train_loader,test_loader=load_data(config)

    model=MNIST_model()
    epoch_num=5

    for epoch in range(epoch_num):
        train_loop=tqdm.tqdm(enumerate(train_loader),total=len(train_loader))
        #test_loop=tqdm.tqdm(enumerate(test_loader),total=len(test_loader))
        train(model, epoch,epoch_num, train_loop)
        test(model,test_loader)