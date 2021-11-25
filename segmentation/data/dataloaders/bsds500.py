#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import errno
import hashlib
import glob
import tarfile
import numpy as np
import torch.utils.data as data
import scipy.io
from PIL import Image


class BSDS500(data.Dataset):
    
    def __init__(self, root='/home/khangt1k25/Code/Contrastive Segmentation/BSDS500',
                 transform=None, overfit=False, split='train', subject=0):
        super(BSDS500, self).__init__()

        self.root = root
        self.transform = transform
        self.subject = int(subject)
        assert(self.subject in [0, 1, 2, 3, 4])
        self.images_dir = os.path.join(self.root, 'images', split)
        self.labels_dir = os.path.join(self.root, 'ground_truth', split)
        
        self.all_images_path = []
        self.all_labels_path = []
        for f in os.listdir(self.images_dir):
            if f.split('.')[1] == 'jpg':
                _image = os.path.join(self.images_dir, f)
                _label = os.path.join(self.labels_dir, f.split('.')[0]+'.mat')
                self.all_images_path.append(_image)
                self.all_labels_path.append(_label)
        # for f in os.listdir(self.labels_dir):
        #     if f.split('.')[1] == 'mat':
        #         _label = os.path.join(self.labels_dir, f)
        #         self.all_labels_path.append(_label)
        

        assert(len(self.all_labels_path) == len(self.all_images_path))
        # Display stats
        print('Number of images: {:d}'.format(len(self.all_images_path)))

    def __getitem__(self, index):
        sample = {}
        sample['image'] = self._load_img(index)
        sample['semseg'] = self._load_label(index)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        sample['meta'] = {'image': str(self.all_images_path[index])}
        return sample

    def __len__(self):
        return len(self.all_images_path)
    
    def _load_img(self, index):
        _img = Image.open(self.all_images_path[index]).convert('RGB')
        return _img
    def _load_label(self, index):
        label = scipy.io.loadmat(self.all_labels_path[index])
        label = label['groundTruth']
        return label[0][self.subject][0][0][0]
        # return label
    

    def __str__(self):
        return 'BSDS500'



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bsd = BSDS500(split='val', subject=1)
    xo = bsd[15]
    # for xo in bsd:
        # yo = xo['semseg']
        # print(type(yo))
        # print(np.unique(yo))
    fig, axes = plt.subplots(2)
    print(np.unique(xo['semseg']))
    axes[0].imshow(xo['image'])
    axes[1].imshow(xo['semseg'])
    plt.show()
    
    
    # xo['image'].show()
    # yo = xo['semseg']
    # print(yo)
    # print(yo.keys())
    # print(xo['image'].size)
    # print(yo['groundTruth'][0][2][0][0][0])
    # print(len(yo['groundTruth'][0]))
    # print(len(yo['groundTruth'][0][0]))
    # print(yo['groundTruth'][0][0][0][0][0])

    # print(yo['groundTruth'][0][0][0][0][1])