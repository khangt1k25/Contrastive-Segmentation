
from os import stat, stat_result
from kornia.geometry import transform
import numpy.random as random
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F
import  kornia as K
import torch.nn as nn





# class MyAugmentation(nn.Module):
#     def __init__(self):
#         super(MyAugmentation, self).__init__()
#         self.k2 = K.augmentation.RandomAffine([-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15], same_on_batch=True, return_transform=False)

#     def forward(self, sample) -> torch.Tensor:
        
#         sample['image'] = self.k2(sample['image'])

#         state_dict = self.k2._params
        
#         sample['sal'] = self.k2(sample['sal'].float(), self.k2._params)
#         sample['sal'] = sample['sal'].int()
        
#         return sample, state_dict

    
#     def forward_with_params(self, sample, state_dict):
             

#         sample['image'] = self.k2(sample['image'], state_dict)

#         sample['sal'] = self.k2(sample['sal'].float(), state_dict).int()
        
#         return sample

class MyAugmentation(nn.Module):
    def __init__(self):
        super(MyAugmentation, self).__init__()
        self.k2 = K.augmentation.RandomAffine([-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15], same_on_batch=True, return_transform=True)

    def forward(self, sample) -> torch.Tensor:
        
        sample['image'], transform = self.k2(sample['image'])

        sample['image'] = sample['image'].squeeze(0)

        state_dict = self.k2._params
        
        sample['sal'], transform = self.k2(sample['sal'].float(), self.k2._params)
        
        sample['sal'] = sample['sal'].squeeze(0).int()
        
        return sample, state_dict, transform

    
    def forward_with_params(self, sample, state_dict):
             

        sample['image'], _ = self.k2(sample['image'], state_dict)
        sample['image'] = sample['image'].squeeze(0)
        sample['sal'], _ = self.k2(sample['sal'].float(), state_dict)

        sample['sal']  = sample['sal'].squeeze(0).int()
        
        return sample
    
    def inverse(self, sample, transform):
        sample['image'] = self.k2.inverse((sample['image'], transform)).squeeze(0)

        sample['sal'] = self.k2.inverse((sample['sal'].float(), transform)).squeeze(0)
        sample['sal']  = sample['sal'].int()
        return sample
    

# class MyAugmentationV1(nn.Module):
#     def __init__(self):
#         super(MyAugmentationV1, self).__init__()
#         self.pipeline = K.augmentation.AugmentationSequential(
#             K.augmentation.RandomAffine([-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15]),
#             data_keys=['input', 'mask'],
#             same_on_batch=True,
#             return_transform=True
#         )
#     def forward(self, sample) -> torch.Tensor:
        
#         output = self.pipeline(sample['image'], sample['sal'].float())


#         sample['image'] = output[0][0]
#         sample['sal'] = output[1][0].int()
#         matrix = output[0][1]
#         return sample, self.pipeline._params

#     def apply(self, sample, matrix):
#         output = self.pipeline(sample['image'],sample['sal'].float, params=matrix)
        

#         # print(output1.shape)
#         # print(output2.shape)
#         return output

# class MyAugmentation(nn.Module):
#     def __init__(self):
#         super(MyAugmentation, self).__init__()
#         self.aff = K.augmentation.RandomAffine(
#             [-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15], return_transform=True, same_on_batch=True
#         )

#     def forward(self, sample) -> torch.Tensor:
        
#         sample['image'], transform = self.aff(sample['image'])

#         state_dict = self.aff._params
        
#         sample['sal'], transform = self.aff(sample['sal'].float(), state_dict)
#         sample['sal'] = sample['sal'].int()
        
#         return sample, transform, state_dict

#     def forward_with_params(self, sample, state_dict):
             

#         sample['image'], transform = self.aff(sample['image'], state_dict)

#         sample['sal'], transform = self.aff(sample['sal'].float(), state_dict)
#         sample['sal'] = sample['sal'].int()
        
#         return sample, transform
