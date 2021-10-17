



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
from utils.common_config import get_model, get_train_dataset, get_inv_transforms, get_eqv_transforms, get_base_transforms
from torchvision.transforms import ToPILImage

from utils.collate import collate_custom
import kornia.augmentation as k_aug
import kornia.geometry.transform as k_trans
import numpy as np
from data.dataloaders.dataset import KorniaDataset, MyDataset


toPIL = ToPILImage()

p = {'train_db_name':'VOCSegmentation', 'train_db_kwargs': {'saliency':'unsupervised_model'}}

base_dataset = get_train_dataset(p, transform=None)
base_transform = get_base_transforms()
inv_list = ['colorjitter', 'gray']
# inv_list = []
eqv_list = ['hflip', 'affine']
# eqv_list = []
inv_transform = get_inv_transforms(inv_list)
eqv_transform = get_eqv_transforms(eqv_list)

train_dataset = MyDataset(base_dataset, base_transform, inv_transform, eqv_transform, inveqv_version=3)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, pin_memory=True, drop_last=True, collate_fn=collate_custom)

for i, batch in enumerate(train_dataloader):
    im_q = batch['query']['image']
    im_k = batch['key']['image']
    im_ie = batch['inveqv']['image']

    sal_q = batch['query']['sal']
    sal_k = batch['key']['sal']
    sal_ie = batch['inveqv']['sal']

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
    
    # ie = deepcopy(im_ie)

    # for j in range(len(eqv_transform)-1, -1, -1):

    #     m = [ele[j] for ele in matrix_eqv]
    #     m = torch.stack(m, dim=0).squeeze()
    #     if(j==len(eqv_transform)-1):
    #         ie = eqv_transform[j].inverse((ie, m),size=size_eqv[0][0])
    #     else:
    #         ie = k_trans.warp_perspective(ie, m, size_eqv[0][0]) 
    # im_q.required_grad = True
    # im_ie.required_grad = False

    # # tmp = im_q.clone().detach().requires_grad_(True)
    # tmp = im_q
    # for j in range(len(eqv_transform)):
    #     m = [ele[j] for ele in matrix_eqv]
    #     m = torch.stack(m, dim=0).squeeze()
    #     # tmp = k_trans.warp_perspective(tmp, m, size_eqv[0][0])
    #     tmp = tmp + 1

    
    loss_1 = (im_q - 3).mean()

    print(loss_1)
    x1_grad = torch.autograd.grad(loss_1, [im_q])
    print(x1_grad)
    # x2_grad = torch.autograd.grad(loss_2, [x2])[0]
    break
    # print(k_transformed.shape)
    # for i in range(3, 6):
    #     # toPIL(im_q[i]).show()
    #     toPIL(im_ie[i]).show()
    #     toPIL(tmp[i]).show()
    #     # toPIL(sal_q[i].float()).show()
    #     # toPIL(im_k[i]).show()
    #     # toPIL(im_ie[i]).show()
    #     # toPIL(sal_ie[i].float()).show()
    # break


