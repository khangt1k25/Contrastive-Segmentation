



from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import matrix
from numpy.core.fromnumeric import size

import torch
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
# import torchvision.transforms as tf

# import kornia.augmentation as k_aug
# import kornia.geometry.transform as k_trans


def show_images(image_block):
    # (row, col, channel, height, width)
    assert len(image_block.shape) == 5, f"image_block.shape={image_block.shape}!"
    row, col, channel, height, width = image_block.shape

    image_block = image_block.permute(0, 1, 3, 4, 2).data.cpu().numpy()

    fig, axes = plt.subplots(row, col)

    for i in range(row):
        for j in range(col):
            axes[i][j].imshow(image_block[i, j])
            axes[i][j].set_aspect('equal')
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    plt.show()
    plt.close(fig)


# ROOT_DIR = "/home/khangt1k25/Code/Contrastive Segmentation/cifar10"
# dataset = CIFAR10(ROOT_DIR, train=True, transform=tf.ToTensor(), download=True)

# loader_1 = DataLoader(dataset, batch_size=7, shuffle=False,
#                       pin_memory=True, num_workers=4)
# loader_1 = iter(loader_1)

# loader_2 = DataLoader(dataset, batch_size=7, shuffle=False,
#                       pin_memory=True, num_workers=4)
# loader_2 = iter(loader_2)

# augmenter_1 = k_aug.RandomAffine(
#     degrees=(-30, -10),
#     translate=(0.15, 0.15),
#     scale=(0.2, 0.5),
#     return_transform=True,
#     same_on_batch=False,
#     p=0.5,
# )

# augmenter_2 = k_aug.RandomHorizontalFlip(p=0.5, return_transform=True)

# augmenter_3 = k_aug.ColorJitter(
#     brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3,
#     return_transform=True   # This one does not have any effect
# )

# # '''
# # ============================== #
# # Check whether we can reuse transform or not

# # CONCLUSION: We can resue transform
# # ============================== #
# for i in range(3):
#     x1, y1 = next(loader_1)
#     x2, y2 = next(loader_2)

#     # Transform batch of images 1
#     # ----------------------------- #
#     # m1_t: Transformation matrix
#     x1_t, m11_t = augmenter_1(x1)
#     # s11_t: (height, width) of the output
#     s11_t = tuple(x1_t.shape[-2:])

#     x1_t, m12_t = augmenter_2(x1_t)
#     s12_t = tuple(x1_t.shape[-2:])

#     x1_t, m13_t = augmenter_3(x1_t)
#     s13_t = tuple(x1_t.shape[-2:])
#     # ----------------------------- #

#     # Apply this transformation to batch of images 2
#     # ----------------------------- #
#     x2_t = k_trans.warp_perspective(x2, m11_t, s11_t)
#     x2_t = k_trans.warp_perspective(x2_t, m12_t, s12_t)

#     # We use different random color jitter
#     # x2_t = augmenter_3(x2_t)

#     # We reuse old color jitter (no thing change)
#     # x2_t = k_trans.warp_perspective(x2_t, m13_t, s13_t)
#     # ----------------------------- #

#     print(f"m13_t:\n{m13_t}")

#     x = torch.stack([x1, x2, x1_t, x2_t], dim=0)
#     show_images(x)
# # '''

# # ============================== #
# # Check whether the gradients of the one that uses kornia and
# # of the one that reuse transforms are equal or not
# # Don't use color jitter here because the results will not be comparable

# # CONCLUSION: If gradients of x1 and x2 will match very well if the gradients are large.
# # However, if the gradients of the two are small, the gradient won't match well.
# # ============================== #
# # torch.set_printoptions(precision=4, threshold=None, linewidth=10000, sci_mode=False)

# for i in range(10):
#     # IMPORTANT: You must ensure that loader_1 and loader_2 has shuffle=False
#     # so the results are comparable
#     x1, y1 = next(loader_1)
#     x2, y2 = next(loader_2)

#     x1.requires_grad = True
#     x2.requires_grad = True

#     # Transform batch of images 1
#     # ----------------------------- #
#     # m1_t: Transformation matrix
#     x1_t, m11_t = augmenter_1(x1)
#     # s11_t: (height, width) of the output
#     s11_t = tuple(x1_t.shape[-2:])

#     x1_t, m12_t = augmenter_2(x1_t)
#     s12_t = tuple(x1_t.shape[-2:])
#     # ----------------------------- #

#     # Apply this transformation to batch of images 2
#     # ----------------------------- #
#     x2_t = k_trans.warp_perspective(x2, m11_t, s11_t)
#     x2_t = k_trans.warp_perspective(x2_t, m12_t, s12_t)
#     # ----------------------------- #

#     x = torch.stack([x1, x2, x1_t, x2_t], dim=0)
#     show_images(x)

#     loss_1 = (x1_t - 3).pow(2).sum(dim=(1, 2, 3)).sqrt().sum(0)
#     loss_2 = (x2_t - 3).pow(2).sum(dim=(1, 2, 3)).sqrt().sum(0)

#     x1_grad = torch.autograd.grad(loss_1, [x1])[0]
#     x2_grad = torch.autograd.grad(loss_2, [x2])[0]

#     print("\n")
#     print(f"x1_grad_check:\n{x1_grad[0, 0, 0:10, 0:10]}")
#     print()
#     print(f"x2_grad_check:\n{x2_grad[0, 0, 0:10, 0:10]}")

# from copy import deepcopy
# from kornia.geometry import transform
# from numpy.core.fromnumeric import size
# from torchvision import transforms
# from data.dataloaders.transforms_v2 import RandomHorizontalFlip, RandomGrayscale, ColorJitter
# from data.dataloaders.dataset import TwoTransformDataset

# from utils.common_config import get_train_dataset,get_base_transformations, get_next_transformations
# from PIL import Image
# import torch
# from utils.collate import collate_custom
# from modules.losses import ConsistencyLoss

# toPIL = transforms.ToPILImage()
# toTensor = transforms.ToTensor()

# # aug = MyAugmentation()

# # aug(sample)

from kornia.geometry import transform
from data.dataloaders.dataset import KorniaDataset
from utils.common_config import get_base_transforms, get_eqv_transforms, get_inv_transforms, get_train_dataset
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
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


#     toPIL(query['sal'].float()).show()


# # train_dataset = TwoTransformDataset(get_train_dataset(p, transform = None), base_transform, next_transform, type_kornia=1, min_area=0.01, max_area=0.99)
# # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_custom)

# # train_dataset = get_train_dataset(p, transform=None)

# hflip = RandomHorizontalFlip(p=1, N=10)
# gray = RandomGrayscale(p=0.5, N=10)
# jitter = ColorJitter(jitter=[0.4, 0.4, 0.4, 0.1], p=1, N=10)

# # sample = train_dataset[0]

# img = Image.open('../examples/pic.jpg')
# sal = torch.rand(size=(224, 224))

# img.show()

# x = {'image': toTensor(img), 'sal': sal}

# x = jitter(0, x)
# x = gray(0, x)
# x= hflip(0, x)

# print(x['image'].squeeze(0).shape)
# print(x['sal'].shape)

# toPIL(x['image'].squeeze(0)).show()

# label = toTensor(img)
# loss = (x['image']-label)
# print(loss)
# loss.backward()

# for i, batch in enumerate(train_dataloader):
#     # Forward pass
#     im_q = batch['query']['image']
#     im_k = batch['key']['image']
#     sal_q = batch['query']['sal']
#     sal_k = batch['key']['sal']

#     state_dict = batch['T']
#     transform = batch['transform']
#     augmented_k = []
#     augmented_sal = []
    
    
#     for i in range(len(state_dict)):

#         sample = {"image": deepcopy(im_k[i]), 'sal': deepcopy(sal_k[i])}
#         new_sample = next_transform.forward_with_params(sample, state_dict[i])
#         augmented_sal.append(new_sample['sal'].squeeze(0))
#         augmented_k.append(new_sample['image'].squeeze(0))

#     augmented_k = torch.stack(augmented_k, dim=0).squeeze(0)
    
#     augmented_k = augmented_k.permute((0, 2, 3, 1))
#     augmented_sal = torch.stack(augmented_sal, dim=0)
   
#     i = 21
    

    
    
    
    # q_selected = im_q.permute((0, 2, 3, 1))

    

    # mask = augmented_sal.unsqueeze(-1)
    
    # x = augmented_k * mask
    # y = q_selected * mask

    # z = (augmented_k-q_selected) * mask
    # # print(z.shape)
    # loss = (z**2).sum(dim=-1)
    # print(loss.shape)
    # print(loss.mean())

    # x =  x.reshape((-1, x.shape[-1]))
    # y =  y.reshape((-1, y.shape[-1]))
    # loss2 = ((x-y)**2).sum(dim=-1).mean()

    # print(loss2)
    # print(x.shape)
    
    
    # print(x.shape)
    # print(y.shape)

    # augmented_k = augmented_k.permute((0, 3, 1, 2))
    # # toPIL(augmented_sal[i].float()).show()
    # # toPIL(augmented_k[i]).show()
    # x = x.permute((0, 3, 1, 2))
    # y = y.permute((0, 3, 1, 2))

    # toPIL(x[i]).show()
    # toPIL(y[i]).show()
   


    

    # return (((x-y) ** 2).sum(dim=-1)).mean()
    # consistency = ConsistencyLoss(type='l2')

    # consistency_loss = consistency(augmented_k, q_selected, mask=augmented_sal)
    # break
# sample = train_dataset[1234]

# key = sample['key']
# query = sample['query']


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
