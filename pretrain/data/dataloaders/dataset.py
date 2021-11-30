#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from numpy.core.fromnumeric import size
import torch.utils.data as data
import random
import warnings
import torchvision
from copy import deepcopy
from torch.nn.functional import interpolate
import kornia.augmentation as k_aug

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



## Our custom dataset
class DatasetKeyQueryInvEqv(data.Dataset):
    def __init__(self, base_dataset, crop_inv_transform, eqv_list_key, eqv_list_query ,downsample_sal=False,
                    scale_factor_sal=0.125, min_area=0.1, max_area=0.99):
        super(DatasetKeyQueryInvEqv, self).__init__()

        self.base_dataset = base_dataset
        self.crop_inv_transform = crop_inv_transform
        self.eqv_list_key = eqv_list_key
        self.eqv_list_query = eqv_list_query

        self.normalize = k_aug.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        # self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
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

            key_sample = self.crop_inv_transform(deepcopy(sample_))
            key_sample, _, _ = self.eqv_transform(self.eqv_list_key, key_sample)

            inveqv_sample = self.crop_inv_transform(deepcopy(sample_))
            query_sample, matrix_eqv, size_eqv = self.eqv_transform(self.eqv_list_query, deepcopy(inveqv_sample))
    

            key_sample['image'] = self.normalize(key_sample['image']).squeeze()
            query_sample['image'] = self.normalize(query_sample['image']).squeeze()
            inveqv_sample['image'] = self.normalize(inveqv_sample['image']).squeeze()

            
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
    
    def eqv_transform(self, pipelines, sample):
        img = sample['image']
        sal = sample['sal']
        matrix_eqv = []
        size_eqv = []
        for eqv in pipelines:
            img, m = eqv(img)
            s = tuple(img.shape[-2:])
            params = eqv._params
            sal, _ = eqv(sal.float(), params)
            matrix_eqv.append(m)
            size_eqv.append(s)
        
        sample['image'] = img.squeeze()
        sample['sal'] = sal.squeeze().long()
        return sample, matrix_eqv, size_eqv   


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss