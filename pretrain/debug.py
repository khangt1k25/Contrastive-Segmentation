
from copy import deepcopy
from utils.common_config import get_train_dataset
import numpy as np
from matplotlib import pyplot as plt
from data.dataloaders.dataset import DatasetKeyQuery, DatasetKeyQueryRandAug
# p['train_db_kwargs']['saliency']
p = {'train_db_name': 'VOCSegmentation', 'overfit': False , 'train_db_kwargs': {'saliency': 'unsupervised_model'}}
base_dataset = get_train_dataset(p, transform=None)


dataset = DatasetKeyQueryRandAug(
                                base_dataset, res=224, 
                                inv_list=['brightness', 'contrast', 'saturation', 'hue' 'blur'],
                                eqv_list=['h_flip', 'v_flip'])

import random
i = random.randint(0, 100)
sample = dataset[i]
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

# print(np.unique(sal_randaug))
# print(np.unique(sal_query))
# print(np.unique(sal_query))

# ok1 = np.transpose(dataset.apply_eqv(i, sample['key']['image']).numpy(), (1, 2, 0))
# ok1 = 255*(ok1 * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))

import torchvision.transforms.functional as TF
import torch 




# ok2 = dataset.apply_randaug(i, deepcopy(sal_key.unsqueeze(0)), is_feat=1)

# ok3 = dataset.apply_randaug(i, torch.randn(size=(3, 32, 224, 224)), is_feat=2)
# print(torch.unique(ok2))
# print(torch.unique(sal_key))
# print(torch.unique(sal_randaug))

ok  = torch.randn(size=(10, 32, 224, 224))
index = torch.randint(0, 20, size=(10, ))
# print(index)

after = dataset.apply_randaug(index, deepcopy(ok), is_feat=2).squeeze()

print(ok-after)
# print(after.shape)

# print(sal_randaug.shape)
# print(randaug.shape)

# TF.to_pil_image(randaug).show()
# TF.to_pil_image(sal_randaug).show()
# TF.to_pil_image(ok2).show()
# TF.to_pil_image(randaug.astype(np.uint8)).show()
# TF.to_pil_image(255*(sample['randaug']['image']* torch.tensor([0.229,0.224,0.225]))+torch.tensor([0.485,0.456,0.406])).show()
# axes[0].imshow(key.astype(np.uint8))
# # axes[1].imshow(query.astype(np.uint8))
# axes[2].imshow(ok2.astype(np.uint8))

# axes[2].imshow(randaug.astype(np.uint8))
# # axes[3].imshow(sal_key)
# # axes[4].imshow(sal_query)
# # axes[5].imshow(sal_randaug)
# plt.show()

# for i, sample in enumerate(dataset):
#     fig, axes = plt.subplots(4)
#     key = np.transpose(sample['key']['image'].numpy(), (1,2,0))
#     key = 255*(key * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
#     query = np.transpose(sample['query']['image'].numpy(), (1,2,0))
#     query = 255*(query * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
#     sal_query = sample['query']['sal']
#     sal_key = sample['key']['sal']
#     axes[0].imshow(key.astype(np.uint8))
#     axes[1].imshow(query.astype(np.uint8))
#     axes[2].imshow(sal_key)
#     axes[3].imshow(sal_query)
#     plt.show()
#     break

# sal_q = torch.Tensor([0])

# cluster_logits = torch.randint(low=0, high=5, size=(1,3))
# pseudo_labels = torch.argmax(cluster_logits, dim=1)
# batch_logits = torch.randint(low=0, high=5, size=(1,3)) # pixels x B
# bank_logits =  torch.randint(low=0, high=5, size=(1,3))         # pixels x negatives
# logits = torch.cat([cluster_logits, batch_logits, bank_logits], dim=1) # pixels x (B+ negatives)

# sal_q += cluster_logits.shape[1]
# print(logits)
# print(logits.shape)
# print(pseudo_labels)
# print(pseudo_labels.shape)
# print(sal_q)
# bsz  = 3


# print(res)
# print(sal_q)
# x_flat = x.reshape(-1)
# print(x_flat)

# sal_q = 

# selected = torch.index_select(x_flat, index=sal_q, dim=0)

# # torch.index_select(q, index=mask_indexes, dim=0)
# selected = selected.reshape(bsz, -1)

# print(selected)
# indices = torch.tensor([0, 2])
# print(torch.index_select(x, 0, indices))
# print(torch.index_select(x, 1, indices))
