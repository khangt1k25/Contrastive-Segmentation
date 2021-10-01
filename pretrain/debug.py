from copy import deepcopy
from random import sample
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from utils.common_config import get_model, get_train_dataset, get_base_transformations, get_next_transformations
import torchvision
from utils.collate import collate_custom
from data.dataloaders.dataset import TwoTransformDataset

toPIL = torchvision.transforms.ToPILImage()

p = {'train_db_name': 'VOCSegmentation', 'train_db_kwargs':{
'saliency': 'supervised_model'}, 'overfit': False, 'backbone':'resnet18','backbone_kwargs': {'dilated':True, 'pretraining':False}, 'head':'deeplab',
'model_kwargs': {'ndim':32, 'head':'linear', 'upsample':True, 'use_classification_head':True, 'use_attention_head':True}}

# inv_list = ['brightness', 'contrast', 'saturation', 'hue', 'gray', 'blur']
# eqv_list = ['h_flip', 'v_flip', 'rotation']



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


