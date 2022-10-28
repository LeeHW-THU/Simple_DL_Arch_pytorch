import yaml
import os
import torch
from tqdm import tqdm
from torchstat import stat


def load_conf(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not os.path.exists('../res'):
        os.mkdir('../res')
    if not os.path.exists('../res/{}'.format(config['Experiment_name'])):
        os.mkdir('../res/{}'.format(config['Experiment_name']))
    with open(os.path.join('../res/{}'.format(config['Experiment_name']), config_path), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print('Config Loaded.')
    return config


def get_optimizer(parameters, config):
    name = config['optimizer']['name']
    optimizer = None
    if name == 'adam':
        lr = config['optimizer']['args']['lr']
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif name == 'SGD':
        lr = config['optimizer']['args']['lr']
        momentum = config['optimizer']['args']['momentum']
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    return optimizer


def print_arch(model, config):
    print('Experiment_name : ', config['Experiment_name'])
    print('Dataset_name : ', config['dataset']['name'])
    print('Dataset_path : ', config['dataset']['root_path'])
    print('Train_BatchSize : {} , Val_BatchSize : {}'.format(
        config['dataset']['train_dataset']['args']['batch_size'],
        config['dataset']['val_dataset']['args']['batch_size']
    ))
    print('Epoch_num={}, Epoch_val={}, Epoch_save={}'.format(
        config['epoch_max'],
        config['epoch_val'],
        config['epoch_save']
    ))
    print('Optimizer : ', config['optimizer']['name'])
    stat(model, (1, 28, 28))
