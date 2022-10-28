import os.path
import torch
from torchvision import datasets, transforms


def load_data(config):
    root_path = config['dataset']['root_path']
    data_name = config['dataset']['name']
    train_loader, test_loader = None, None
    # for DIV2K data
    if data_name == 'DIV2K':
        train_loader, test_loader = div2k_dataloader(root_path, config)
    # for mnist data
    if data_name == 'MNIST':
        train_loader, test_loader = mnist_dataloader(root_path, config)
    # f0r CIFAR10 data
    if data_name == 'CIFAR10':
        train_loader, test_loader = cifar10_dataloader(root_path, config)

    return train_loader, test_loader


def div2k_dataloader(root_path, config):

    '''
    if mode=='pair':
    HR_path=os.path.join(root_path,'DIV2K_train_HR')
    LR_path=os.path.join(root_path,'DIV2K_train_LR')
    print('HR load from :', HR_path)
    print('LR load from :', LR_path)
    elif mode=='single':
    pass
    '''

    return None, None


def mnist_dataloader(root_path, config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root_path, train=True, download=True,
                       transform=transform),
        batch_size=config['dataset']['train_dataset']['args']['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root_path, train=False, download=True,
                       transform=transform),
        batch_size=config['dataset']['val_dataset']['args']['batch_size'], shuffle=True)

    return train_loader, test_loader


def cifar10_dataloader(root_path, config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root_path, train=True, download=True,
                       transform=transform),
        batch_size=config['dataset']['train_dataset']['args']['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root_path, train=False, download=True,
                       transform=transform),
        batch_size=config['dataset']['val_dataset']['args']['batch_size'], shuffle=True)

    return train_loader, test_loader
