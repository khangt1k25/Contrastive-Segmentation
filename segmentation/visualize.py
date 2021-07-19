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
from utils.kmeans_utils import save_embeddings_to_disk, eval_kmeans
from termcolor import colored
import torchvision.transforms as transforms
from termcolor import colored

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
        if k.startswith('module.model_q'):
            new_state[k.rsplit('module.model_q.')[1]] = v
    msg = model.load_state_dict(new_state, strict=False)
    print(msg)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    from data.dataloaders.pascal_voc import VOC12
    val_transforms = get_val_transformations()
    print(val_transforms)
    val_dataset = VOC12(split='val', transform=val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)

    true_val_dataset = VOC12(split='val', transform=None)
    print(colored('Val samples %d' %(len(true_val_dataset)), 'yellow'))



    i = 16
    sample = val_dataset[i]
    p['embedding_dir'] = '/content/drive/MyDrive/UCS_local/kmeans_result_3/kmeans_VOCSegmentation_supervised_saliency/embeddings'
    filename = os.path.join(p['embedding_dir'], sample['meta']['image'] + '.npy')
    embedding = np.load(filename)
    semseg = sample['semseg']
    image = sample['image']
  

    embedding = torch.from_numpy(embedding)
    embedding = embedding.unsqueeze(0)
    
 
    image = torch.cat([image, embedding], dim=0)
    
    image = image.permute(1, 2, 0).contiguous()
    




    def normalize(x):
      """
      Normalize a list of sample image data in the range of 0 to 1
      : x: List of image data.  The image shape is (N, N, 3)
      : return: Numpy array of normalized data
      """
      return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
    new = normalize(image.numpy())
    plt.imsave("./origin.jpg", new[:,:,:-1])
    plt.imsave("./segmented.jpg", new)
    plt.imsave("./mask.jpg", new[:,:,-1])


    

    

if __name__ == "__main__":
    main()
