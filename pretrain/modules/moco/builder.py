#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#
# MoCo implementation based upon https://arxiv.org/abs/1911.05722
# 
# Pixel-wise contrastive loss based upon our paper


import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.common_config import get_model
from modules.losses import BalancedCrossEntropyLoss

class ContrastiveModel(nn.Module):
    def __init__(self, p):
        """
        p: configuration dict
        """
        
        super(ContrastiveModel, self).__init__()

        self.K = p['moco_kwargs']['K'] 
        self.m = p['moco_kwargs']['m'] 
        self.T = p['moco_kwargs']['T']

        # create the model 
        self.model_q = get_model(p)
        self.model_k = get_model(p)

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.dim = p['model_kwargs']['ndim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # balanced cross-entropy loss
        self.bce = BalancedCrossEntropyLoss(size_average=True)

              

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, sal_q, sal_k, num_negatives = 5):
        """
        Input:
            images: a batch of images (B x 3 x H x W) 
            sal: a batch of saliency masks (B x H x W)
        Output:
            logits, targets
        """
        batch_size = im_q.size(0)
        self.avg2d = self.avg2d.to(sal_q.device)

        q, q_bg = self.model_q(im_q)                      # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        
        q_flat = q.permute((0, 2, 3, 1))                  # queries: B x H x W x dim 
        q_flat = torch.reshape(q_flat, [-1, self.dim])    # queries: pixels x dim

        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)
    
        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            tmp = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            tmp = tmp.view(-1)
            mask_indexes = torch.nonzero((tmp)).view(-1).squeeze()
            tmp = torch.index_select(tmp, index=mask_indexes, dim=0) // 2

        # compute key prototypes
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, _ = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            
            # prototypes k
            k_flat = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_k = sal_k.reshape(batch_size, -1, 1).type(k.dtype) # B x H.W x 1
            prototypes_foreground = torch.bmm(k_flat, sal_k).squeeze() # B x dim
            prototypes = nn.functional.normalize(prototypes_foreground, dim=1)        
        
        # Compute local contrastive logits, labels
        
        # ver1: hard computational  
        # l_logits = []
        # kernel = 3
        # for i in range(q.shape[0]):
        #     # Working with each image
        #     indexes = torch.nonzero((sal_q[i]).view(-1)).squeeze()      # indexes: opixels
        #     q_i = q[i].view(-1, self.dim)                               # q_i:  HW x dim
        #     object_i = q_i[indexes]                                     # object_i: opixels x dim
            
        #     local_i = F.avg_pool2d(q[i], kernel_size=kernel, stride=1, padding=1)
        #     local_i = local_i.view(-1, self.dim)[indexes]

        #     # with torch.no_grad():
        #     #     local_k = F.avg_pool2d(k)                                     
        #     #     local_k = local_k.view(-1, self.dim)[indexes]               # local_k: opixels x dim
        #     #     k_i = k[i].view(-1, self.dim)                               # k_i:   HW x dim


        #     # local positive
        #     similarity = torch.matmul(object_i, local_i.T)
        #     mask = torch.eye(object_i.shape[0], dtype=torch.bool)
        #     local_positive = similarity[mask].view(object_i.shape[0], -1)


        #     # local negative
        #     neg = torch.matmul(q_i, q_i.T)
        #     local_negative = torch.zeros(size=(object_i.shape[0], num_negatives), dtype=torch.float)
        #     a = torch.arange(q_i.shape[0]).float()
        #     weight = torch.ones_like(a)
        #     for j in range(indexes.shape[0]):
        #         # random pixels exclude current pixel
        #         cur_index = indexes[j].item()
        #         weight[cur_index] = 0.
        #         neg_indexes = a[torch.multinomial(weight, num_samples=num_negatives)].long()
        #         local_negative[j] = neg[cur_index][neg_indexes]
        #         weight[cur_index] = 1.

        #     local_logits = torch.cat([local_positive, local_negative], dim=1)
        #     l_logits.append(local_logits)


        # l_logits = torch.cat(l_logits, dim=0)
        # l_labels = torch.zeros(l_logits.shape[0])  
        # end ver1
      
        # ver2: lightly computational 
        self.H, self.W = q.shape[2], q.shape[3]
        x_step =  [1, -1, 0, 0]
        y_step =  [0, 0, 1, -1]
        l_logits = []
        for i in range(q.shape[0]):
            # Working with each image
            q_i = q[i]                         
            #k_i = k[i]
            indexes = torch.nonzero(sal_q[i])  #[opixels, 2]
            indexes_range = torch.nonzero((sal_q[i]).view(-1)).squeeze()       #[opixels]
          
            for j in range(indexes.shape[0]):
                  row, col = indexes[j].cpu().numpy()
                  # get a positive
                  pos_row, pos_col = row, col
                  while True:
                      tmp = random.randint(0, 3)
                      pos_row, pos_col = pos_row+x_step[tmp], pos_col+y_step[tmp]
                      if pos_row >= 0 and pos_row < self.H and pos_col >=0 and pos_col < self.W:
                        break
                      pos_row, pos_col = row, col
                  
                  positive = torch.Tensor([torch.dot(q_i[:,row, col], q_i[:, pos_row, pos_col])]).to(sal_q.device)
                  
                  # get negatives
                  q_i_flat = q_i.view(q_i.shape[0], -1)
                  a = torch.arange(self.H*self.W).float()
                  weight = torch.ones_like(a)
                  cur_index = indexes_range[j]
                  weight[cur_index] = 0.
                  neg_indexes = a[torch.multinomial(weight, num_samples=num_negatives)].long()

                  negative = torch.matmul(q_i[:,row, col], q_i_flat[:,neg_indexes]).to(sal_q.device) 

                  logit = torch.cat([positive, negative])
                  
                
                  l_logits.append(logit)

        l_logits = torch.stack(l_logits)
        l_labels = torch.zeros(l_logits.shape[0])      
        
        # end ver2 


        # q: pixels x dim
        # k: pixels x dim
        # prototypes_k: proto x dim
        q = torch.index_select(q_flat, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q_flat, prototypes.t())   # shape: pixels x proto
        negatives = self.queue.clone().detach()          # shape: dim x negatives
        l_mem = torch.matmul(q_flat, negatives)          # shape: pixels x negatives (Memory bank)
        logits = torch.cat([l_batch, l_mem], dim=1)      # pixels x (proto + negatives)

        # apply temperature
        logits /= self.T
        l_logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes) 

        return logits, tmp, l_logits, l_labels, sal_loss



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



