
from os import stat, stat_result
import numpy.random as random
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F
import  kornia as K
import torch.nn as nn



class KorniaColorJitter(nn.Module):
    def __init__(self, jitter):
        super(KorniaColorJitter, self).__init__()
        self.jitter = K.augmentation.ColorJitter(jitter[0], jitter[1], jitter[2], jitter[3])


    def __call__(self, sample):
        sample['image'] = self.jitter(sample['image']) 
        return sample

    def __str__(self):
        return 'KorniaColorJitter'


class KorniaRandomAffine(nn.Module):
    def __init__(self, affine):
        super(KorniaRandomAffine, self).__init__()
        self.affine = K.augmentation.RandomAffine(affine[0], affine[1], affine[2], affine[3])

    def __call__(self, sample):
        sample['image'] = self.affine(sample['image'])
        sample['sal'] = sample['sal'].unsqueeze(0)
        sample['sal'] = self.affine(sample['sal'].float())
        sample['sal'] = sample['sal'].squeeze(0)
        sample['sal'] = sample['sal'].int()
        return sample

    def __str__(self):
        return 'KorniaRandomAffine'

class MyAugmentation(nn.Module):
    def __init__(self):
        super(MyAugmentation, self).__init__()
        # we define and cache our operators as class members
        #self.k1 = K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8)
        self.k2 = K.augmentation.RandomAffine([-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15])
    
    def forward(self, sample) -> torch.Tensor:
        #sample['image'] = self.k2(self.k1(sample['image']))
        
        sample['image'] = self.k2(sample['image'])

        state_dict = self.k2._params
        sample['sal'] = self.k2(sample['sal'].float(), self.k2._params).int()
        
        return sample, state_dict
    
    def forward_with_params(self, sample, state_dict):
        
      

        sample['image'] = self.k2(sample['image'], state_dict)

        sample['sal'] = self.k2(sample['sal'].float(), state_dict).int()
        
        return sample;