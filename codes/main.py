import argparse
import os.path

from units import *
from dataloader import load_data
from model import *
from train import *
from test import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, default='./options.yaml', type=str, help='Select GPUs')
    args = parser.parse_args()

    # Set options
    config = load_conf(args.config)

    if config['mode']['name'] == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['mode']['ids'])
        print('Now is GPU mode. DEVICES : ', config['mode']['ids'])
    else:
        print('Now is CPU mode')

    # Set DataLoader
    train_loader, test_loader = load_data(config)

    # Define model
    model = MNIST_model()

    # Print
    print_arch(model, config)

    for epoch in range(config['epoch_max']+1):
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        train(model, epoch, config, train_loop)
        if epoch % config['epoch_val'] == 0:
            test(model, config, test_loader)
        if epoch % config['epoch_save'] == 0:
            torch.save(model.state_dict(), os.path.join('../res/{}'.format(config['Experiment_name']), 'Epoch{}.pt'.format(epoch)))

    torch.save(model.state_dict(), os.path.join('../res/{}'.format(config['Experiment_name']), 'FinalEpoch.pt'))