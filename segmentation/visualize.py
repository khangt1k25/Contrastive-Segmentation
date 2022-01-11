#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.config import create_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_dataloader,\
                                get_model
from termcolor import colored
import torchvision.transforms as transforms
from termcolor import colored
from utils.visualization import visualize_sample, visualize_sample_with_prediction, visualize_sample_with_saved_prediction
from tqdm import tqdm
# Parser
parser = argparse.ArgumentParser(description='Fully-supervised segmentation')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # CUDNN
    # print(colored('Set CuDNN benchmark', 'blue')) 
    # torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    from data.dataloaders.pascal_voc import VOC12
    # val_transforms = get_val_transformations()
    # print(val_transforms)
    # val_dataset = VOC12(split='val', transform=val_transforms)
    # val_dataloader = get_val_dataloader(p, val_dataset)

    true_val_dataset = VOC12(split='val', transform=None)
    print(colored('Val samples %d' %(len(true_val_dataset)), 'yellow'))

    num_samples = 20

    for i in tqdm(range(num_samples)):
      sample = true_val_dataset[i]
      visualize_sample_with_saved_prediction(p, sample, os.path.join(p['visualize_dir'], 'vis{}.png'.format(str(i))))
    

if __name__ == "__main__":
    main()
