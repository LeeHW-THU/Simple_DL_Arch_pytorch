import os.path
import torch
from torchvision import datasets, transforms

def load_data(config):
    root_path=config['train_dataset']['dataset']['args']['root_path']
    mode=config['train_dataset']['dataset']['args']['mode']
    batch_size=config['train_dataset']['dataset']['args']['batch_size']
    data_name=config['train_dataset']['dataset']['args']['name']
    # for DIV2K data
    if data_name=='DIV2K':
        train_loader, test_loader=DIV2K_dataloader(root_path,mode,batch_size)
    # for mnist data
    if data_name=='MNIST':
        train_loader, test_loader=MNIST_dataloader(root_path,batch_size)
    return train_loader, test_loader
def DIV2K_dataloader(root_path,mode,batch_size):
    if mode=='pair':
        HR_path=os.path.join(root_path,'DIV2K_train_HR')
        LR_path=os.path.join(root_path,'DIV2K_train_LR')
        print('HR load from :', HR_path)
        print('LR load from :', LR_path)
    elif mode=='single':
        pass
def MNIST_dataloader(root_path,batch_size):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root_path, train=True, download=True,
        transform=transform),batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root_path, train=True, download=True,
        transform=transform),batch_size=batch_size, shuffle=True)


    return train_loader, test_loader