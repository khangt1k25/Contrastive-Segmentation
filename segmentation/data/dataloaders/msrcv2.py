#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import errno
import hashlib
import glob
import zipfile
import numpy as np
import torch.utils.data as data
from data.util.mypath import Path
from data.util.google_drive import download_file_from_google_drive
from utils.utils import mkdir_if_missing
from PIL import Image
import random

class MSRC(data.Dataset):

    GOOGLE_DRIVE_ID = '1VnOq18ae8jYd71lyXCgRF6GyaZ3vTqpn'

    FILE = 'MSRCv2.zip'
    SEG_LABELS_LIST = [
    {"id": -1, "name": "void",       "rgb_values": [0,   0,    0]},
    {"id": -1,  "name": "horse",      "rgb_values": [128, 0,    128]},
    {"id": -1,  "name": "mountain",   "rgb_values": [64,  0,    0]},
    {"id": 0,  "name": "building",   "rgb_values": [128, 0,    0]},
    {"id": 1,  "name": "grass",      "rgb_values": [0,   128,  0]},
    {"id": 2,  "name": "tree",       "rgb_values": [128, 128,  0]},
    {"id": 3,  "name": "cow",        "rgb_values": [0,   0,    128]},
    # {"id": 4,  "name": "horse",      "rgb_values": [128, 0,    128]},
    {"id": 4,  "name": "sheep",      "rgb_values": [0,   128,  128]},
    {"id": 5,  "name": "sky",        "rgb_values": [128, 128,  128]},
    # {"id": 7,  "name": "mountain",   "rgb_values": [64,  0,    0]},
    {"id": 6,  "name": "airplane",   "rgb_values": [192, 0,    0]},
    {"id": 7,  "name": "water",      "rgb_values": [64,  128,  0]},
    {"id": 8, "name": "face",       "rgb_values": [192, 128,  0]},
    {"id": 9, "name": "car",        "rgb_values": [64,  0,    128]},
    {"id": 10, "name": "bicycle",    "rgb_values": [192, 0,    128]},
    {"id": 11, "name": "flower",     "rgb_values": [64,  128,  128]},
    {"id": 12, "name": "sign",       "rgb_values": [192, 128,  128]},
    {"id": 13, "name": "bird",       "rgb_values": [0,   64,   0]},
    {"id": 14, "name": "book",       "rgb_values": [128, 64,   0]},
    {"id": 15, "name": "chair",      "rgb_values": [0,   192,  0]},
    {"id": 16, "name": "road",       "rgb_values": [128, 64,   128]},
    {"id": 17, "name": "cat",        "rgb_values": [0,   192,  128]},
    {"id": 18, "name": "dog",        "rgb_values": [128, 192,  128]},
    {"id": 19, "name": "body",       "rgb_values": [64,  64,   0]},
    {"id": 20, "name": "boat",       "rgb_values": [192, 64,   0]}]
    
    
    def __init__(self, root=Path.db_root_dir('MSRCv2'),
                 transform=None, overfit=False, split='train', download=False):
        super(MSRC, self).__init__()

        self.CATEGORY_NAMES = ['building', 'grass', 'tree', 'cow', 'sheep', 'sky',
        'airplane', 'water', 'face', 'car', 'bicycle', 'flower', 'sign', 'bird', 'book', 'chair', 'road', 
        'cat', 'dog', 'body', 'boat']
        
        
        self.root = root
        self.transform = transform
        self.split = split
        if download:
            self._download()
        self.names_dir = os.path.join(self.root, 'Meta', self.split+'.txt')
        self.images_dir = os.path.join(self.root, 'Images')
        self.labels_dir = os.path.join(self.root, 'GroundTruth')
        
        self.images_path = []
        self.labels_path = []
        

        with open(self.names_dir, 'r') as f:
            names = f.read().splitlines()

        random.shuffle(names)

        for f in names:
            _image = os.path.join(self.images_dir, f + '.bmp')
            _label = os.path.join(self.labels_dir, f + '_GT.bmp')
            self.images_path.append(_image)
            self.labels_path.append(_label)

        assert(len(self.images_path) == len(self.labels_path))
     
        self.ignore_classes = [-1]
        
        # Display stats
        print('Number of images: {:d}'.format(len(self.images_path)))
        
        
    def __getitem__(self, index):    
        sample = {}
        sample['image'] = self._load_img(index)
        sample['semseg'] = self._load_label(index)
        
        if self.transform is not None:
            sample = self.transform(sample)

        sample['meta'] = {'im_size': (sample['image'].shape[0], sample['image'].shape[1]),
                          'image_file': self.images_path[index],
                          'image': os.path.basename(self.images_path[index]).split('.')[0]}
        
        return sample

    def __len__(self):
        
        return len(self.images_path)

    def _load_img(self, index):
        
        _img = np.array(Image.open(self.images_path[index]).convert('RGB'))
        
        return _img
    
    def _load_label(self, index):
        
        _semseg = np.array(Image.open(self.labels_path[index]).convert('RGB'))
        
        

        _semseg_label = _semseg[..., 0]
        for label in self.SEG_LABELS_LIST:

            mask = np.all(_semseg == label['rgb_values'], axis=2)
            _semseg_label[mask] = label['id']

        for ignore_class in self.ignore_classes:
            _semseg_label[_semseg_label == ignore_class] = 255
        
        return _semseg_label

    def __str__(self):
        return 'MSRCv2'
    def get_class_names(self):
        return self.CATEGORY_NAMES
    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        print('\nExtracting zip file')
        import zipfile
        with zipfile.ZipFile(self.FILE, 'r') as zip_ref:
            zip_ref.extractall(Path.db_root_dir())
        
        print('Done!')
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bsd = MSRC(split='train')
    xo = bsd[16]
    print(xo['image'].size)
    fig, axes = plt.subplots(2)
    print(np.unique(xo['semseg']))
    axes[0].imshow(xo['image'])
    axes[1].imshow(xo['semseg'])
    plt.show()
    
