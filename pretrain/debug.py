import torch

a = torch.zeros([])


print(a.long())
# from copy import deepcopy
# import matplotlib.pyplot as plt
# from numpy import matrix, mod
# from numpy.core.fromnumeric import size

# import torch
# import torch.nn as nn
# import numpy as np
# from matplotlib import pyplot as plt
# from torch.utils import data
# from torchvision import transforms
# from utils.common_config import get_model, get_train_dataset, get_inv_transforms, get_eqv_transforms, get_base_transforms
# from torchvision.transforms import ToPILImage

# from utils.collate import collate_custom
# import kornia.augmentation as k_aug
# import kornia.geometry.transform as k_trans
# import numpy as np
# from data.dataloaders.dataset import MyDataset
# from modules.models import Filter, PredictionHead

# toPIL = ToPILImage()

# p = {'train_db_name':'VOCSegmentation', 'train_db_kwargs': {'saliency':'unsupervised_model'}, 
# 'backbone':'resnet18', 'backbone_kwargs': {'dilated':True, 'pretraining':False}, 'model_kwargs': {
#     'ndim': 32, 'head':'linear', 'upsample':True, 'use_classification_head': True}, 'head':'deeplab'
# }

# base_dataset = get_train_dataset(p, transform=None)
# base_transform = get_base_transforms()
# inv_list = ['colorjitter', 'gray']
# # inv_list = []
# eqv_list = ['hflip', 'affine']
# # eqv_list = []
# inv_transform = get_inv_transforms(inv_list)
# eqv_transform = get_eqv_transforms(eqv_list)

# train_dataset = MyDataset(base_dataset, base_transform, inv_transform, eqv_transform, inveqv_version=1)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, pin_memory=True, drop_last=True, collate_fn=collate_custom)


# model = get_model(p)
# prediction_head = PredictionHead(dim=32)

# mask_head = Filter()

# salmap = torch.Tensor([[0, 1, 1], [0, 1, 1], [0, 0, 1]])
# salmap = salmap.unsqueeze(0)

# after = mask_head(salmap)
# print(salmap.shape)

# print(after.shape)
# print(salmap)
# print(after)
# print(after * salmap)

# # for i, batch in enumerate(train_dataloader):
# #     im_q = batch['query']['image']
# #     im_k = batch['key']['image']
# #     im_ie = batch['inveqv']['image']

# #     sal_q = batch['query']['sal']
# #     sal_k = batch['key']['sal']
# #     sal_ie = batch['inveqv']['sal']

# #     matrix_eqv = batch['matrix']
# #     size_eqv = batch['size']
    

# #     q, bg_q = model(im_q)
# #     q = nn.functional.normalize(q, dim=1)

# #     ie, bg_ie = model(im_ie)
# #     ie = nn.functional.normalize(ie, dim=1)

# #     pred = prediction_head(q)
# #     print(pred[0,:,0,0])
# #     pred = nn.functional.normalize(pred, dim=1)
# #     print(q.shape)
# #     print(pred.shape)
# #     print(pred[0,:,0,0])
#     # print(ie[0,:,0,0])   
    
#     # # ie = deepcopy(im_ie)
#     # toPIL(q[0,:3,:,:]).show()
#     # toPIL(ie[0,:3,:,:]).show()
#     # for j in range(len(eqv_list)):
#     #     m = [ele[j] for ele in matrix_eqv]
#     #     m = torch.stack(m, dim=0).squeeze()
#     # ie = k_trans.warp_perspective(ie, m, size_eqv[0][0])
#     # toPIL(ie[0,:3,:,:]).show()

#     # # print(k_transformed.shape)
#     # for i in range(3, 6):
#     #     toPIL(im_q[i]).show()
#     #     toPIL(ie[i]).show()
#         # toPIL(ie[i]).show()
#         # toPIL(sal_ie[i].float()).show()
#         # toPIL(sal_q[i].float()).show()
#         # toPIL(im_k[i]).show()
#         # toPIL(im_ie[i]).show()
#         # toPIL(sal_ie[i].float()).show()
#     # toPIL(ie[0]).show()
#     # pred = prediction_head(q)
#     # print(pred[0,:,0,0])
    
#     # toPIL(pred[0,:3,:,:]).show()
#     # pred = nn.functional.normalize(pred, dim=1)
#     # toPIL(pred[0,:3,:,:]).show()
#     # print(q[0,:,0,0])
#     # print(q[0,:,0,0])
#     # print(ie[0,:,0,0])
#     # print(pred[0,:,0,0])
#     # print(im_ie[0,:,0,0])    
#     # print(im_q[0,:,0,0])
#     # affine = k_aug.RandomAffine(
#     #             degrees=(10, 30),
#     #             translate=(0.15, 0.15),
#     #             scale=(0.5, 1),
#     #             return_transform=True,
#     #             same_on_batch=False,
#     #             p=0.5
#     #         )
    
#     # ie = deepcopy(im_ie)
#     # sal_ie = sal_ie.unsqueeze(1)
#     # # print(sal_ie.shape)
#     # # print(ie.shape) 
#     # for j in range(len(eqv_transform)-1, -1, -1):

#     #     m = [ele[j] for ele in matrix_eqv]
#     #     m = torch.stack(m, dim=0).squeeze()
#     #     if(j==len(eqv_transform)-1):
#     #         ie = eqv_transform[j].inverse((ie, m),size=size_eqv[0][0])
#     #         sal_ie = eqv_transform[j].inverse((sal_ie.float(), m),size=size_eqv[0][0])
#     #     else:
#     #         ie = k_trans.warp_perspective(ie, m, size_eqv[0][0]) 
#     #         sal_ie = k_trans.warp_perspective(sal_ie.float(), m, size_eqv[0][0]).long() 

#     # sal_ie = sal_ie.squeeze()

#     # im_q.required_grad = True
#     # im_ie.required_grad = False

#     # # tmp = im_q.clone().detach().requires_grad_(True)
#     # tmp = im_q
#     # for j in range(len(eqv_transform)):
#     #     m = [ele[j] for ele in matrix_eqv]
#     #     m = torch.stack(m, dim=0).squeeze()
#     #     # tmp = k_trans.warp_perspective(tmp, m, size_eqv[0][0])
#     #     tmp = tmp + 1

    


    
    
    # break


