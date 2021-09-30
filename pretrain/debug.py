from copy import deepcopy
from random import sample
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from utils.common_config import get_train_dataset, get_base_transformations, get_next_transformations
import torchvision
from utils.collate import collate_custom
from data.dataloaders.dataset import TwoTransformDataset

toPIL = torchvision.transforms.ToPILImage()

p = {'train_db_name': 'VOCSegmentation', 'train_db_kwargs':{
'saliency': 'supervised_model'}, 'overfit': False}


# inv_list = ['brightness', 'contrast', 'saturation', 'hue', 'gray', 'blur']
# eqv_list = ['h_flip', 'v_flip', 'rotation']


train_dataset = get_train_dataset(p)
base_transform = get_base_transformations()
next_transform = get_next_transformations()

my_dataset = TwoTransformDataset(train_dataset, base_transform, next_transform, 
                                downsample_sal=False) 
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_custom)

for i in range(10):
    batch = my_dataset[i]

    # print(batch.keys())

    

    key = batch['key']
    query = batch['query']
    T = batch['T']
    transform = batch['transform']


    # toPIL(query['image']).show()
    toPIL(query['sal'].float()).show()
    # toPIL(key['image']).show()
    # toPIL(key['sal'].float()).show()


    # next = next_transform.forward_with_params(deepcopy(query), T)
    inv = next_transform.inverse(deepcopy(key), transform)



    # toPIL(next['image']).show()
    # toPIL(next['sal'].float()).show()

    # print(torch.unique(next['sal']-key['sal']))
    print(torch.unique(inv['sal']-query['sal']))


    # toPIL(inv['image']).show()
    toPIL(inv['sal'].float()).show()


# batch_size = 32
# dim = 28
# q = torch.randn(size=(batch_size, dim, 224, 224))
# sal_q = torch.randn(size=(batch_size, 224, 224))
# sal_q = torch.round(sal_q)


# q_mean = q.reshape(batch_size, dim, -1) # B x dim x H.W
# sal_q_flat = sal_q.reshape(batch_size, -1, 1).type(q.dtype) # B x H.W x 1
# q_mean = torch.bmm(q_mean, sal_q_flat).squeeze() # B x dim
# q_mean = nn.functional.normalize(q_mean, dim=1)  

# prototypes = torch.randn(size=(batch_size, dim))
# print(q_mean.shape)

# l_batch = torch.matmul(q_mean, prototypes.t())

# print(l_batch.shape)


