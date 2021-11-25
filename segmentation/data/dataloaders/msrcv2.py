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
from data.util.mypath import Path
from data.util.google_drive import download_file_from_google_drive
from utils.utils import mkdir_if_missing
from PIL import Image
import random

class MSRC(data.Dataset):
    GOOGLE_DRIVE_ID = '1itp-JFJU062V8GCeGNFoduDE4eW5FueW'

    FILE = 'MSRCv2.tar.xz'

    
    def __init__(self, root=Path.db_root_dir('MSRCv2'),
                 transform=None, overfit=False, split='train', download=False):
        super(MSRC, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split

        self.names_dir = os.path.join(self.root, 'Meta', self.split+'.txt')
        self.images_dir = os.path.join(self.root, 'Images')
        self.labels_dir = os.path.join(self.root, 'GroundTruth')
        
        self.images_path = []
        self.labels_path = []
        if download:
            self._download()

        with open(self.names_dir, 'r') as f:
            names = f.read().splitlines()

        random.shuffle(names)

        for f in names:
            _image = os.path.join(self.images_dir, f + '.bmp')
            _label = os.path.join(self.labels_dir, f + '_GT.bmp')
            self.images_path.append(_image)
            self.labels_path.append(_label)

        assert(len(self.images_path) == len(self.labels_path))
     

        # Display stats
        print('Number of images: {:d}'.format(len(self.images_path)))
        
        
    def __getitem__(self, index):    
        sample = {}
        sample['image'] = self._load_img(index)
        sample['semseg'] = self._load_label(index)
        
        if self.transform is not None:
            sample = self.transform(sample)
        sample['meta'] = {'image_path': str(self.images_path[index]), "semseg_path":str(self.labels_path[index])}
        
        return sample

    def __len__(self):
        
        return len(self.images_path)

    def _load_img(self, index):
        
        _img = np.array(Image.open(self.images_path[index]).convert('RGB'))
        
        return _img
    
    def _load_label(self, index):
        
        _semseg = np.array(Image.open(self.labels_path[index]).convert('RGB'))
        
        return _semseg

    def __str__(self):
        return 'MSRCv2'

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
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bsd = MSRC(split='train')
    xo = bsd[16]
    # for xo in bsd:
        # yo = xo['semseg']
        # print(type(yo))
        # print(np.unique(yo))
    print(xo['image'].size)
    fig, axes = plt.subplots(2)
    print(np.unique(xo['semseg']))
    axes[0].imshow(xo['image'])
    axes[1].imshow(xo['semseg'])
    # print(xo['semseg'])
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