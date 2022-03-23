#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
from random import seed
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from utils.utils import *
from utils.kmeans_utils import *
from utils.config import create_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_transformations_invariance ,get_val_dataloader,\
                                get_model, get_val_transformations_invariance
from termcolor import colored
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Fully-unsupervised segmentation')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--num_seeds', default=5, type=int,
                    help='number of seeds during kmeans')
args = parser.parse_args()

def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print(model)
    model = model.cuda()
    
    # Load pre-trained weights
    state_dict = torch.load(p['pretraining'], map_location='cpu')
        # State dict follows our lay-out
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('model_q'):
            new_state[k.rsplit('model_q.')[1]] = v
    msg = model.load_state_dict(new_state, strict=False)
    print(msg)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    from data.dataloaders.pascal_voc import VOC12
    val_transforms_invariances = get_val_transformations_invariance()
    print(val_transforms_invariances)
    val_dataset = VOC12(split='val', transform=val_transforms_invariances)
    # print(val_dataset[0])
    val_dataloader = get_val_dataloader(p, val_dataset)

    val_transforms = get_val_transformations()
    print(val_transforms)
    true_val_dataset = VOC12(split='val', transform=val_transforms)
    true_val_dataloader = get_val_dataloader(p, true_val_dataset)
    

    # Kmeans Clustering
    n_clusters = 21
    results_miou = []
    results_acc = []
    best = 0.0
    for i in range(args.num_seeds):
        eval_result = eval_kmeans_pixel(p, val_dataloader, true_val_dataloader, model, n_clusters=n_clusters, seed=1234+i, verbose=True)
        if eval_result['mean_iou'] > best:
            best = eval_result['mean_iou']
        results_miou.append(eval_result['mean_iou'])    
        results_acc.append(eval_result['overall_precision (pixel accuracy)'])
    
    print(colored('Average mIoU is %2.1f' %(np.mean(results_miou)), 'green'))
    print(colored('STD mIoU is %2.1f' %(np.std(results_miou)), 'green'))
    print(colored('Average Acc is %2.1f' %(np.mean(results_acc)), 'green'))
    print(colored('STD Acc is %2.1f' %(np.std(results_acc)), 'green'))
    
if __name__ == "__main__":
    main()
