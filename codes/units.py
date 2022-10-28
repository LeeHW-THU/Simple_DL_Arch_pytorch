import yaml
import os
import torch
from torch.backends import cudnn
def load_conf(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not os.path.exists('./res'):
        os.mkdir('./res')
    if not os.path.exists('./res/{}'.format(config['name'])):
        os.mkdir('./res/{}'.format(config['name']))
    with open(os.path.join('./res/{}'.format(config['name']),config_path), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print('Config Loaded.')
    return config

def set_envs(config):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        print('CUDA_VISIBLE_DEVICES : ',config['gpu'])
    else:
        print('Now is CPU mode')