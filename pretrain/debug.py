from select import select
from numpy import indices
import torch
from PIL import Image
import os
import torch.nn as nn


x = torch.rand(10, 32)

maxval, label = x.topk(k=1, dim=1)

print(maxval.shape)
print(label.shape)

mask = maxval.ge(0.9).float()

print(mask.shape)

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
