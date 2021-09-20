from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
# from matplotlib import pyplot as plt
# from torch.utils import data
# from utils.common_config import get_train_dataset
# import torchvision
# from utils.collate import collate_custom

# p = {'train_db_name': 'VOCSegmentation', 'train_db_kwargs':{
# 'saliency': 'supervised_model'}, 'overfit': False}


# inv_list = ['brightness', 'contrast', 'saturation', 'hue', 'gray', 'blur']
# eqv_list = ['h_flip', 'v_flip', 'rotation']


# train_dataset = get_train_dataset(p, inv_list=inv_list, eqv_list=eqv_list) 
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_custom)


# index, batch = train_dataset[1]

# key = batch['key']
# query = batch['query']


# key['image'].show()
# key['sal'].show()
# query['image'].show()
# query['sal'].show()
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


from modules.models import AttentionHead

z = torch.randn(size=(2, 64, 224, 224))
sal_z = torch.rand(size=(2, 224, 224))
sal_z = torch.round(sal_z)
head = AttentionHead(dim=64)

out = head(z, sal_z)

print(out.shape)