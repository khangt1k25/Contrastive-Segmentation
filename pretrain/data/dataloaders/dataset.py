#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from kornia.augmentation import base
from numpy.core.fromnumeric import size
import torch.utils.data as data
import random
import warnings
import torchvision
from copy import deepcopy
from torch.nn.functional import interpolate
import kornia.augmentation as k_aug
import kornia.geometry.transform as k_trans

class Dataset(data.Dataset):
    def __init__(self, base_dataset, train_transform, downsample_sal=False,
                    scale_factor_sal=0.125, min_area=0.01, max_area=0.99):
        super(Dataset, self).__init__()
        self.base_dataset = base_dataset
        self.train_transform = train_transform
        self.downsample_sal = downsample_sal
        
        if isinstance(scale_factor_sal, float):
            self.scale_factor_sal = (scale_factor_sal, scale_factor_sal)
        else:
            self.scale_factor_sal = scale_factor_sal

        self.min_area = min_area
        self.max_area = max_area

    def __len__(self):
        return len(self.base_dataset) 

    def __getitem__(self, index):
        sample_ = self.base_dataset.__getitem__(index)
        count = 0
        
        while True:
            if count > 1: # Warning
                #warnings.warn('Need to re-apply transform for image {}'.format(sample['meta']['image']))
                pass

            if count > 2: # Failed to load image two times in a row. Try a different one.
                #warnings.warn('Try loading a different image. Failed to load {}'.format(sample['meta']['image']))
                sample_ = self.base_dataset.__getitem__(random.randint(0, self.__len__()-1))
                count = 100
 
            sample = self.train_transform(deepcopy(sample_))
                           
            if self.downsample_sal: # Downsample
                sample['sal'] = interpolate(sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
            area = sample['sal'].float().sum() / sample['sal'].numel()
            
            if area < self.max_area and area > self.min_area: # Ok. Foreground/Background has proper ratio.
                return sample

            else:
                count += 1 # Try again. Areas of foreground/background to small.


class DatasetKeyQuery(data.Dataset):
    def __init__(self, base_dataset, transform, downsample_sal=False,
                    scale_factor_sal=0.125, min_area=0.1, max_area=0.99):
        super(DatasetKeyQuery, self).__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.downsample_sal = downsample_sal
        
        if isinstance(scale_factor_sal, float):
            self.scale_factor_sal = (scale_factor_sal, scale_factor_sal)
        else:
            self.scale_factor_sal = scale_factor_sal

        self.min_area = min_area
        self.max_area = max_area

    def __len__(self):
        return len(self.base_dataset) 

    def __getitem__(self, index):
        sample_ = self.base_dataset.__getitem__(index)
        count = 0
        
        while True:
            if count > 1: # Warning
                #warnings.warn('Need to re-apply transform for image {}'.format(sample['meta']['image']))
                pass

            if count > 2: # Failed to load image two times in a row. Try a different one.
                #warnings.warn('Try loading a different image. Failed to load {}'.format(sample['meta']['image']))
                sample_ = self.base_dataset.__getitem__(random.randint(0, self.__len__()-1))
                count = 100

            key_sample = self.transform(deepcopy(sample_))
            query_sample = self.transform(deepcopy(sample_))
                           
            if self.downsample_sal: # Downsample
                key_sample['sal'] = interpolate(key_sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
                query_sample['sal'] = interpolate(query_sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
            key_area = key_sample['sal'].float().sum() / key_sample['sal'].numel()
            query_area = query_sample['sal'].float().sum() / query_sample['sal'].numel()
            
            if key_area < self.max_area and key_area > self.min_area and query_area < self.max_area and query_area > self.min_area: # Ok. Foreground/Background has proper ratio.
                return {'key': key_sample, 'query': query_sample}

            else:
                count += 1 # Try again. Areas of foreground/background to small.




        img = sample['image']
        sal = sample['sal']
        matrix_eqv = []
        size_eqv = []
        for eqv in self.eqv_list:
            # print(img.shape)
            img, m = eqv(img)
            s = tuple(img.shape[-2:])
            params = eqv._params
            sal, _ = eqv(sal.float(), params)
        
            matrix_eqv.append(m)
            size_eqv.append(s)
        
        sample['image'] = img.squeeze()
        sample['sal'] = sal.squeeze()
        return sample, matrix_eqv, size_eqv



## Our custom dataset
class MyDataset(data.Dataset):
    def __init__(self, base_dataset, base_transform, inv_list, eqv_list, downsample_sal=False,
                    scale_factor_sal=0.125, min_area=0.1, max_area=0.99):
        super(MyDataset, self).__init__()

        self.base_dataset = base_dataset
        self.base_transform = base_transform
        self.inv_list = inv_list
        self.eqv_list = eqv_list


        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.downsample_sal = downsample_sal
        
        if isinstance(scale_factor_sal, float):
            self.scale_factor_sal = (scale_factor_sal, scale_factor_sal)
        else:
            self.scale_factor_sal = scale_factor_sal

        self.min_area = min_area
        self.max_area = max_area

    def __len__(self):
        return len(self.base_dataset) 

    def __getitem__(self, index):
        sample_ = self.base_dataset.__getitem__(index)
        count = 0
        
        while True:
            if count > 1: # Warning
                #warnings.warn('Need to re-apply transform for image {}'.format(sample['meta']['image']))
                pass

            if count > 2: # Failed to load image two times in a row. Try a different one.
                #warnings.warn('Try loading a different image. Failed to load {}'.format(sample['meta']['image']))
                sample_ = self.base_dataset.__getitem__(random.randint(0, self.__len__()-1))
                count = 100

            key_sample = self.base_transform(deepcopy(sample_))
         
            query_sample = self.base_transform(deepcopy(sample_))


            
            
                
            inveqv_sample, _, _ = self.inv_transform(deepcopy(query_sample))

            query_sample, matrix_eqv, size_eqv = self.eqv_transform(query_sample)
    

            key_sample['image'] = self.normalize(key_sample['image'])
            query_sample['image'] = self.normalize(query_sample['image'])
            inveqv_sample['image'] = self.normalize(inveqv_sample['image'])

            if self.downsample_sal: # Downsample
                key_sample['sal'] = interpolate(key_sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
                query_sample['sal'] = interpolate(query_sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
                inveqv_sample['sal'] = interpolate(inveqv_sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
            
            key_area = key_sample['sal'].float().sum() / key_sample['sal'].numel()
            query_area = query_sample['sal'].float().sum() / query_sample['sal'].numel()
            inveqv_area = inveqv_sample['sal'].float().sum() / inveqv_sample['sal'].numel()
            
            if key_area < self.max_area and key_area > self.min_area and query_area < self.max_area and query_area > self.min_area and inveqv_area < self.max_area and inveqv_area > self.min_area: # Ok. Foreground/Background has proper ratio.
                return {'key': key_sample, 'query': query_sample, 'inveqv': inveqv_sample, 'matrix': matrix_eqv, 'size': size_eqv}


            else:
                count += 1 # Try again. Areas of foreground/background to small.
    
    
    def inv_transform(self, sample):
        img = sample['image']
        sal = sample['sal']
        matrix_inv = []
        size_inv = []
        for inv in self.inv_list:
            img, m = inv(img)
            matrix_inv.append(m)
            size_inv.append(tuple(img.shape[-2:]))
        
        sample['image'] = img.squeeze()
        
        return sample, matrix_inv, size_inv

    def eqv_transform(self, sample):
        img = sample['image']
        sal = sample['sal']
        matrix_eqv = []
        size_eqv = []
        for eqv in self.eqv_list:
    
            img, m = eqv(img)
            s = tuple(img.shape[-2:])
            params = eqv._params
            sal, _ = eqv(sal.float(), params)
            matrix_eqv.append(m)
            size_eqv.append(s)
        
        sample['image'] = img.squeeze()
        sample['sal'] = sal.squeeze().long()
        return sample, matrix_eqv, size_eqv   

