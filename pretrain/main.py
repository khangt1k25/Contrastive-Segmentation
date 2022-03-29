#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import builtins
import os
from pickletools import optimize
import sys
from termcolor import colored

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataloaders.dataset import DatasetKeyQuery, DatasetKeyQueryRandAug

from modules.moco.builder import ContrastiveModel

from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_train_dataloader, get_optimizer, adjust_learning_rate, get_randaug_transformations

from utils.train_utils import train
from utils.logger import Logger
from utils.collate import collate_custom


# Parser
parser = argparse.ArgumentParser(description='Main function')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--nvidia-apex', action='store_true',
                    help='Use mixed precision')

# Distributed params
parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                            help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

def main():
    args = parser.parse_args()
    main_worker(0, args=args)


def main_worker(gpu, args):
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)

    # Check gpu id
    args.gpu = gpu
    p['gpu'] = gpu
    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
        

    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = ContrastiveModel(p)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    
    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model.parameters())
    print(optimizer)

    amp = None

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    # train_transform = get_train_transformations()
    # print(train_transform)
    # randaug_transform = get_randaug_transformations(m=10)
    
    # base_dataset, res=224, min_area=0.01, max_area=0.99, inv_list=[], eqv_list=[])
    # train_dataset = DatasetKeyQueryRandAug(get_train_dataset(p, transform = None), train_transform, randaug_transform,
    #                             downsample_sal=not p['model_kwargs']['upsample'])
    train_dataset = DatasetKeyQueryRandAug(get_train_dataset(p, transform = None), res=224, inv_list=['brightness', 'contrast', 'saturation', 'hue' 'gray'],
                                            eqv_list=['h_flip'])
    #                             downsample_sal=not p['model_kwargs']['upsample'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=p['train_batch_size'], shuffle=False,
                    num_workers=p['num_workers'], pin_memory=True, drop_last=True, collate_fn=collate_custom)
    print(colored('Train samples %d' %(len(train_dataset)), 'yellow'))
    print(colored(train_dataset, 'yellow'))
    
    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(p['checkpoint'], map_location=loc)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()
    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        eval_train = train(p, train_dataloader, model, 
                                    optimizer, epoch)

        # eval_train = train_mc(p, train_dataloader, model,
        #                             optimizer, epoch)
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, 
                    p['checkpoint'])


if __name__ == "__main__":
    main()
