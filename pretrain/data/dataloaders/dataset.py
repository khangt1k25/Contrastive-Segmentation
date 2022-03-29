#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.utils.data as data
import random
import warnings

from copy import deepcopy
from torch.nn.functional import interpolate
from data.dataloaders.custom_transforms import *
from data.dataloaders.randaugment import RandAugment



class Dataset(data.Dataset):
    def __init__(self, base_dataset, train_transform, downsample_sal=False,
                    scale_factor_sal=0.125, min_area=0.1, max_area=0.99):
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


class DatasetKeyQueryRandAug(data.Dataset):
    def __init__(self, base_dataset, res=224, min_area=0.01, max_area=0.99, inv_list=[], eqv_list=[]):
        super(DatasetKeyQueryRandAug, self).__init__()
        self.base_dataset = base_dataset
        self.res = res
        self.min_area = min_area
        self.max_area = max_area
        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.N = len(self.base_dataset)
        self.init_transforms()

    def __len__(self):
        return len(self.base_dataset) 

    def init_transforms(self):
        # self.transform_base = RandomResizedCrop(size=self.res, scale=(0.2, 1)) 
        self.transform_base = RandomResizedCrop(size=self.res, scale=(0.2, 1))
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.4, p=0.8, N=self.N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.4, p=0.8, N=self.N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.4, p=0.8, N=self.N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=self.N) for _ in range(2)]      # Control this later (NOTE)   
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=self.N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=self.N) for _ in range(2)]
        
        # Transforms for equivariance
        self.random_horizontal_flip = RandomHorizontalFlip(N=self.N)
        self.random_vertical_flip   = RandomVerticalFlip(N=self.N)
        self.horizontal_tensor_flip = RandomHorizontalTensorFlip(N=self.N, p_ref=self.random_horizontal_flip.p_ref, plist=self.random_horizontal_flip.plist)
        self.vertical_tensor_flip = RandomVerticalTensorFlip(N=self.N, p_ref=self.random_vertical_flip.p_ref, plist=self.random_vertical_flip.plist)


        # RandAugment
        self.randAugment = RandAugment(N=self.N) 
        # Tensor and normalize transform. 
        self.transform_tensor = TensorTransform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def transform_inv(self, index, image, ver):
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        
        return image

    def transform_eqv(self, index, sample):
        
        if 'h_flip' in self.eqv_list:
            sample  = self.random_horizontal_flip(index, sample)
        if 'v_flip' in self.eqv_list:
            sample = self.random_vertical_flip(index, sample)

        return sample


    def __getitem__(self, index):
        sample_ = self.base_dataset.__getitem__(index)
        # count = 0
        while True:
                        

            sample = self.transform_base(index, deepcopy(sample_))

            # Key 
            key_sample = deepcopy(sample)
            key_sample['image'] = self.transform_inv(index, key_sample['image'], ver=0)
            key_sample = self.transform_tensor(key_sample)

            # Query
            query_sample = deepcopy(sample)
            query_sample['image'] = self.transform_inv(index, query_sample['image'], ver=1)
            query_sample = self.transform_eqv(index, query_sample)
            query_sample = self.transform_tensor(query_sample)

            # Randaug
            randaug_sample = deepcopy(sample)
            randaug_sample = self.randAugment(index, randaug_sample)
            randaug_sample = self.transform_tensor(randaug_sample)

            return {'key': key_sample, 'query': query_sample, 'randaug': randaug_sample, 'index': index}

            # key_area = key_sample['sal'].float().sum() / key_sample['sal'].numel()
            # query_area = query_sample['sal'].float().sum() / query_sample['sal'].numel()
            # randaug_area = randaug_sample['sal'].float().sum() / randaug_sample['sal'].numel()

            # print(key_area, query_area, randaug_area)
        
            # if key_area < self.max_area and key_area > self.min_area and query_area < self.max_area and query_area > self.min_area and randaug_area < self.max_area and randaug_area > self.min_area: # Ok. Foreground/Background has proper ratio.
            #     return {'key': key_sample, 'query': query_sample, 'randaug': randaug_sample, 'index': index}
            
    def apply_eqv(self, index, feat):
        
        # index = index.cpu().numpy()
        if 'h_flip' in self.eqv_list:
            sample  = self.horizontal_tensor_flip(index, feat)
        if 'v_flip' in self.eqv_list:
            sample = self.vertical_tensor_flip(index, feat)
        return sample

    def apply_randaug(self, index, feat, is_feat=1):
        
        # index = index.cpu().numpy()
        
        feat = self.randAugment.apply(index, feat, is_feat=is_feat)
        
        return feat




if __name__=='__main__':
    import numpy as np
    from matplotlib import pyplot as plt
   

    p = {'train_db_name': 'VOCSegmentation', 'overfit': False}
    from pascal_voc import VOCSegmentation
    base_dataset = VOCSegmentation(root=Path.db_root_dir(p['train_db_name']),
                            saliency=p['train_db_kwargs']['saliency'],
                            transform=None)
    

    dataset = DatasetKeyQueryRandAug(
                                    base_dataset, res=224, 
                                    inv_list=['brightness', 'contrast', 'saturation', 'hue' 'blur'],
                                    eqv_list=['h_flip', 'v_flip'])

    for i, sample in enumerate(dataset):
        fig, axes = plt.subplots(6)
        key = np.transpose(sample['key']['image'].numpy(), (1,2,0))
        key = 255*(key * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
        
        query = np.transpose(sample['query']['image'].numpy(), (1,2,0))
        query = 255*(query * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
        
        randaug = np.transpose(sample['randaug']['image'].numpy(), (1,2,0))
        randaug = 255*(randaug * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))

        sal_query = sample['query']['sal']
        sal_key = sample['key']['sal']
        sal_randaug = sample['randaug']['sal']

        axes[0].imshow(key.astype(np.uint8))
        axes[1].imshow(query.astype(np.uint8))
        axes[2].imshow(randaug.astype(np.uint8))
        axes[3].imshow(sal_key)
        axes[4].imshow(sal_query)
        axes[5].imshow(sal_randaug)
        plt.show()
