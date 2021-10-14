



from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import matrix
from numpy.core.fromnumeric import size

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from utils.common_config import get_model, get_train_dataset, get_base_transformations, get_next_transformations
import torchvision
from utils.collate import collate_custom
import kornia.augmentation as k_aug
import kornia.geometry.transform as k_trans
import numpy as np

toPIL = ToPILImage()

p = {'train_db_name':'VOCSegmentation', 'train_db_kwargs': {'saliency':'unsupervised_model'}}

base_dataset = get_train_dataset(p, transform=None)
base_transform = get_base_transforms()
inv_list = ['colorjitter', 'gray']
eqv_list = ['hflip', 'vflip', 'affine']
inv_transform = get_inv_transforms(inv_list)
eqv_transform = get_eqv_transforms(eqv_list)

train_dataset = KorniaDataset(base_dataset, base_transform, inv_transform, eqv_transform, inveqv_version=2)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=True, collate_fn=collate_custom)

for i, batch in enumerate(train_dataloader):
    im_q = batch['query']['image']
    im_k = batch['key']['image']
    sal_q = batch['query']['sal']
    sal_k = batch['key']['sal']
    matrix_eqv = batch['matrix']
    size_eqv = batch['size']
    # print(im_q.shape)
    # print(im_k.shape)
    # print(sal_q.shape)
    # print(sal_k.shape)

    
    # FOR loop 

    # k_transformed = []
    # for i in range(len(matrix_eqv)):
    #     tmp = deepcopy(im_k[i]).unsqueeze(0)
    #     for j in range(len(eqv_transform)):
    #         tmp = k_trans.warp_perspective(tmp, matrix_eqv[i][j], size_eqv[i][j])
    #         # print(tmp.shape)
    #     tmp = tmp.squeeze()
    #     k_transformed.append(tmp)
    # k_transformed = torch.randn(size=(32, 5, 224, 224))
    

    # affine = k_aug.RandomAffine(
    #             degrees=(10, 30),
    #             translate=(0.15, 0.15),
    #             scale=(0.5, 1),
    #             return_transform=True,
    #             same_on_batch=False,
    #             p=0.5
    #         )
    sal_k_transformed  = deepcopy(sal_k).unsqueeze(1)
    k_transformed = deepcopy(im_k)

    for j in range(len(eqv_transform)-1, -1, -1):

        m = [ele[j] for ele in matrix_eqv]
        m = torch.stack(m, dim=0).squeeze()
        if(j==len(eqv_transform)-1):
            k_transformed = eqv_transform[j].inverse((k_transformed, m),size=size_eqv[0][0])
            sal_k_transformed = eqv_transform[j].inverse((sal_k_transformed, m),size=size_eqv[0][0])
        else:
            k_transformed = k_trans.warp_perspective(k_transformed, m, size_eqv[0][0]) 
            sal_k_transformed = k_trans.warp_perspective(sal_k_transformed, m, size_eqv[0][0]) 
        

    # print(k_transformed.shape)
    for i in range(3, 6):
        # toPIL(im_q[i]).show()
        # toPIL(im_k[i]).show()
        toPIL(k_transformed[i]).show()
        toPIL(sal_k_transformed[i].float()).show()
        # toPIL(sal_k[i].float()).show()
    break
# for i in range(3, 6, 1):
#     sample = train_dataset[i]
#     key = sample['key']
#     query = sample['query']

#     print(key['image'].shape)


#     # toPIL(key['image']).show()
#     toPIL(query['image']).show()
#     # toPIL(key['sal'].float()).show()


p = {'train_db_name': 'VOCSegmentation', 'train_db_kwargs':{
'saliency': 'supervised_model'}, 'overfit': False, 'backbone':'resnet18','backbone_kwargs': {'dilated':True, 'pretraining':False}, 'head':'deeplab',
'model_kwargs': {'ndim':32, 'head':'linear', 'upsample':True, 'use_classification_head':True, 'use_attention_head':True}}

# # train_dataset = TwoTransformDataset(get_train_dataset(p, transform = None), base_transform, next_transform, type_kornia=1, min_area=0.01, max_area=0.99)
# # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_custom)

# # train_dataset = get_train_dataset(p, transform=None)


base_transform = get_base_transformations()
next_transform = get_next_transformations()
train_dataset = get_train_dataset(p)
my_dataset = TwoTransformDataset(train_dataset, base_transform, next_transform, 
                                downsample_sal=False) 
train_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=2, collate_fn=collate_custom)


model = get_model(p)

for i, batch in enumerate(train_dataloader):
    # Forward pass
    im_q = batch['query']['image']
    im_k = batch['key']['image']
    sal_q = batch['query']['sal']
    sal_k = batch['key']['sal']
    state_dict = batch['T']
    transform = batch['transform']

    q, bg_q, q_mask = model(im_q)
    k, bg_k, k_mask = model(im_k)
    fw_q = []
    fw_sal = []
    inv_q = []
    inv_sal = []
    for i in range(len(state_dict)):
        sample1 = {"image":q[i].clone(), "sal":sal_q[i].clone()}
        new_sample1 = next_transform.forward_with_params(sample1, state_dict[i])

        sample2 = {"image": new_sample1['image'].clone(), "sal": new_sample1['sal'].clone()}

        new_sample2 = next_transform.inverse(new_sample1, transform[i])

        fw_q.append(new_sample1['image'].squeeze(0))
        fw_sal.append(new_sample2['sal'].squeeze(0))

        inv_q.append(new_sample2['image'].squeeze(0))
        inv_sal.append(new_sample2['sal'].squeeze(0))

    
    fw_q = torch.stack(fw_q, dim=0).squeeze(0)
    # fw_q = fw_q.permute((0, 2, 3, 1))
    fw_sal = torch.stack(fw_sal, dim=0)
    
    inv_q = torch.stack(inv_q, dim=0).squeeze(0)
    # inv_q = inv_q.permute((0, 2, 3, 1))
    inv_sal = torch.stack(inv_sal, dim=0)

    print(q.shape)
    print(fw_q.shape)
    print(inv_q.shape)
    print(q-inv_q)
    print(sal_q-inv_sal)
    
    # inverse_k = []
    # inverse_sal = []
    # for i in range(len(transform)):
    #     sample = {"image": k[i].detach().clone(), 'sal': bg_q[i].detach().clone()}
    #     new_sample = next_transform.inverse(sample, transform[i])
        

    #     inverse_k.append(new_sample['image'].squeeze(0))
    #     inverse_sal.append(new_sample['sal'].squeeze(0))

    # inverse_k = torch.stack(inverse_k, dim=0).squeeze(0)
    # inverse_k = inverse_k.permute((0, 2, 3, 1))                  
    # inverse_sal = torch.stack(inverse_sal, dim=0)

    # q_selected = q.permute((0, 2, 3, 1))

    # print(inverse_k.shape)
    # print(q_selected.shape)
    # print(inverse_k[0]-q_selected[0])

    # inveqv_loss = self.cons(q_selected, inverse_k, mask=inverse_sal)
        
    # print(im_q.shape)
    # print(im_k.shape)

    # toPIL(im_q[0]).show()
    # toPIL(im_k[0]).show() 
    # sample1 = {"image":im_q[0], "sal":sal_q[0]}

    # next = next_transform.forward_with_params(deepcopy(sample1), state_dict[0])
    # toPIL(next['image']).show()

    # sample2 = {"image": im_k[0], "sal":sal_k[0]}
    # inv = next_transform.inverse(deepcopy(sample2), transform[0])
    # toPIL(inv['image']).show()
    # break

# for i in range(10):
#     batch = my_dataset[i]

#     # print(batch.keys())

    

#     key = batch['key']
#     query = batch['query']
#     T = batch['T']
#     transform = batch['transform']


#     # toPIL(query['image']).show()
#     toPIL(query['sal'].float()).show()
#     # toPIL(key['image']).show()
#     # toPIL(key['sal'].float()).show()


#     # next = next_transform.forward_with_params(deepcopy(query), T)
#     inv = next_transform.inverse(deepcopy(key), transform)



#     # toPIL(next['image']).show()
#     # toPIL(next['sal'].float()).show()

#     # print(torch.unique(next['sal']-key['sal']))
#     print(torch.unique(inv['sal']-query['sal']))


#     # toPIL(inv['image']).show()
#     toPIL(inv['sal'].float()).show()


# print(torch.unique(key['sal']))

# toPIL(key['image']).show(title='key')
# # toPIL(key['sal'].float()).show()
# toPIL(query['image']).show(title='query')
# # toPIL(query['sal'].float()).show()


# next = next_transform.forward_with_params(deepcopy(query), state_dict=sample['T'])

# inv = next_transform.inverse(deepcopy(key), transform=sample['transform'])

# toPIL(inv['image']).show(title='inv')
# toPIL(next['image']).show()

# toPIL(inv['sal'].float()).show()

# toPIL(next['sal'].float()).show()
