
from os import stat, stat_result
from kornia.geometry import transform
import numpy.random as random
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import  kornia as K
import torch.nn as nn






class MyAugmentation(nn.Module):
    def __init__(self):
        super(MyAugmentation, self).__init__()
        self.colorJiter = K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8, same_on_batch=True)
        self.randGrayScale = K.augmentation.RandomGrayscale(p=0.2, same_on_batch=True)
        self.randHorizontalFlip = K.augmentation.RandomHorizontalFlip(p=0.5, same_on_batch=True, return_transform=True)
        self.randAffine = K.augmentation.RandomAffine([-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15], p=0.5, same_on_batch=True, return_transform=True)

    def forward(self, sample) -> torch.Tensor:
        
        sample['image'] = self.colorJiter(sample['image']).squeeze(0)
        sample['image'] = self.randGrayScale(sample['image']).squeeze(0)

        sample['image'], transform_1 = self.randHorizontalFlip(sample['image'])
        sample['image'] = sample['image'].squeeze(0)

        sample['image'], transform_2 = self.randAffine(sample['image'])
        
        sample['image'] = sample['image'].squeeze(0)
        
        state_dict_1 = self.randHorizontalFlip._params
        state_dict_2 = self.randAffine._params    
        
        sample['sal'], transform_1 = self.randHorizontalFlip(sample['sal'].float(), self.randHorizontalFlip._params)
        sample['sal'], transform_2 = self.randAffine(sample['sal'].float(), self.randAffine._params)
        
        sample['sal'] = sample['sal'].squeeze(0).int()
        
        return sample, [state_dict_1, state_dict_2], [transform_1, transform_2]

    
    def forward_with_params(self, sample, state_dict):
             
        sample['image'], _ = self.randHorizontalFlip(sample['image'], state_dict[0])
        sample['image'] = sample['image'].squeeze(0)

        sample['image'], _ = self.randAffine(sample['image'], state_dict[1])
        sample['image'] = sample['image'].squeeze(0)

        sample['sal'], _ = self.randHorizontalFlip(sample['sal'].float(), state_dict[0])
        sample['sal']  = sample['sal'].squeeze(0)
        sample['sal'], _ = self.randAffine(sample['sal'].float(), state_dict[1])
        sample['sal']  = sample['sal'].squeeze(0).int()
        
        return sample
    
    def inverse(self, sample, transform):
        sample['image'] = self.randHorizontalFlip.inverse(sample['image']).squeeze(0)
        sample['image'] = self.randAffine.inverse((sample['image'], transform[1])).squeeze(0)

        sample['sal'] = self.randHorizontalFlip.inverse(sample['sal'].float()).squeeze(0)
        sample['sal'] = self.randAffine.inverse((sample['sal'].float(), transform[1])).squeeze(0)
        sample['sal']  = sample['sal'].int()
        return sample
    

