from select import select
from numpy import indices
import torch
from PIL import Image
import os
import torch.nn as nn
from modules.losses import FocalLoss



predicted = torch.randn((16, 20))
predicted = torch.softmax(predicted, dim=1)

target  = torch.randint(low=0, high=19, size=(16,))

fc_loss = FocalLoss(gamma=2)

loss = fc_loss(predicted, target)
ce_loss = nn.functional.cross_entropy(predicted, target, reduction='mean')

print(loss)
print(ce_loss)
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
