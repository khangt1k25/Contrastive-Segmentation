#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#
# MoCo implementation based upon https://arxiv.org/abs/1911.05722
# 
# Pixel-wise contrastive loss based upon our paper


from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision

from utils.common_config import get_model, get_pHead
from modules.losses import BalancedCrossEntropyLoss, ConsistencyLoss, AttentionLoss
import kornia.augmentation as k_aug
import kornia.geometry.transform as k_trans

class ContrastiveModel(nn.Module):
    def __init__(self, p):
        """
        p: configuration dict
        """
        
        super(ContrastiveModel, self).__init__()

        self.K = p['moco_kwargs']['K'] 
        self.m = p['moco_kwargs']['m'] 
        self.T = p['moco_kwargs']['T']
        self.p = p

        # create the model 
        self.model_q = get_model(p)
        self.model_k = get_model(p)
        self.pHead = get_pHead(p)



        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient




        # create the queue
        self.dim = p['model_kwargs']['ndim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        

        # additional loss
        
        self.bce = BalancedCrossEntropyLoss(size_average=True)
        self.att = AttentionLoss()
        self.cons = ConsistencyLoss(type=p['inveqv_kwargs']['type'])


        
        
        
        


    
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

    def forward(self, im_q, im_k, sal_q, sal_k, im_ie, sal_ie, matrix_eqv, size_eqv, dataloader):
        """
        Input:
            images: a batch of images (B x 3 x H x W) 
            sal: a batch of saliency masks (B x H x W)
        Output:
            logits, targets, local_logits, local_targets
        """
        

        batch_size, channel, H, W = im_q.size()


        q, bg_q = self.model_q(im_q)                      # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        flat_q = q.permute((0, 2, 3, 1))                  
        flat_q = torch.reshape(flat_q, [-1, self.dim])    # queries: pixels x dim


        if self.p['loss_coeff']['spatial'] > 0:
            vertical = (q[:,:,:-1, :] - q[:,:,1:,:]) * (sal_q[:,:,:-1,:])
            horizontal = (q[:,:,:,:-1] - q[:,:,:,1:])* (sal_q[:,:,:,:-1])
            spatial_loss  = vertical.mean() + horizontal.mean()


        # anchor mean
        if self.p['loss_coeff']['mean'] > 0:
            q_mean = q.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_q_flat = sal_q.reshape(batch_size, -1, 1).type(q.dtype) # B x H.W x 1
            q_mean = torch.bmm(q_mean, sal_q_flat).squeeze() # B x dim
            q_mean = nn.functional.normalize(q_mean, dim=1) 

        '''
        Compute saliency loss
        '''
        sal_loss = self.bce(bg_q, sal_q)
      
       
        '''
        Prepare mask_indexes with both query and key size.
        '''
        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            tmp = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            tmp = tmp.view(-1)
            mask_indexes = torch.nonzero((tmp)).view(-1).squeeze()
            tmp = torch.index_select(tmp, index=mask_indexes, dim=0) // 2

        '''
        Prepare prototypes in key size and apply transform 
        '''
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, bg_k = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)       
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            
            # prototypes k
            k_flat = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_k_flat = sal_k.reshape(batch_size, -1, 1).type(k.dtype) # B x H.W x 1
            prototypes_foreground = torch.bmm(k_flat, sal_k_flat).squeeze() # B x dim
            prototypes = nn.functional.normalize(prototypes_foreground, dim=1)   
            

            # apply transform 
            if self.p['loss_coeff']['inveqv'] > 0:
                ie, _ = self.model_k(im_ie)
                ie = nn.functional.normalize(ie, dim=1)   
                if self.p['inveqv_version'] == 1:
                    # forward reuse
                    for j in range(len(dataloader.dataset.eqv_list)):
                        m = [ele[j] for ele in matrix_eqv]
                        m = torch.stack(m, dim=0).squeeze()
                        ie = k_trans.warp_perspective(ie, m, size_eqv[0][0])
                    
                elif self.p['inveqv_version'] == 2: 
                    # inverse reuse
                    sal_ie = sal_ie.unsqueeze(1) 
                    for j in range(len(dataloader.dataset.eqv_list)-1, -1, -1):

                        m = [ele[j] for ele in matrix_eqv]
                        m = torch.stack(m, dim=0).squeeze()
                        if(j==len(dataloader.dataset.eqv_list)-1):
                            ie = dataloader.dataset.eqv_list[j].inverse((ie, m),size=size_eqv[0][0])
                            sal_ie = dataloader.dataset.eqv_list[j].inverse((sal_ie.float(), m),size=size_eqv[0][0])
                        else:
                            ie = k_trans.warp_perspective(ie, m, size_eqv[0][0]) 
                            sal_ie = k_trans.warp_perspective(sal_ie.float(), m, size_eqv[0][0]).long() 
                    
                    sal_ie = sal_ie.squeeze()
        '''
        Compute Consistency loss
        '''
        if self.p['loss_coeff']['inveqv'] > 0:
            if self.p['inveqv_version'] == 1:
                pred = self.pHead(q)
                pred = pred.permute((0, 2, 3, 1))
                ie = ie.permute((0, 2, 3, 1))
                inveqv_loss = self.cons(pred, ie, mask=sal_q)

            elif self.p['inveqv_version'] == 2:
                pred = self.pHead(q)
                pred = pred.permute((0, 2, 3, 1))
                ie = ie.permute((0, 2, 3, 1))
                inveqv_loss = self.cons(pred, ie, mask=sal_q)
        else:
            inveqv_loss = 0.


        '''
        Compute Object Contrastive loss 
        '''

        anchor = torch.index_select(flat_q, index=mask_indexes, dim=0)
        l_batch = torch.matmul(anchor, prototypes.t())   # shape: pixels x proto
        negatives = self.queue.clone().detach()          # shape: dim x negatives
        l_mem = torch.matmul(anchor, negatives)          # shape: pixels x negatives (Memory bank)
        logits = torch.cat([l_batch, l_mem], dim=1)      # pixels x (proto + negatives)
        
        '''
        Compute superpixel contrastive loss 
        '''

        if self.p['loss_coeff']['mean'] > 0:
            l_positive = torch.matmul(q_mean, prototypes.t())
            l_negative = torch.matmul(q_mean, negatives)
            mean_logits = torch.cat([l_positive, l_negative], dim=1)
            mean_labels = torch.arange(mean_logits.shape[0]).to(q.device)
            mean_labels = mean_labels.long()
        else:
            mean_logits = 0.
            mean_labels = 0.
        
        # apply temperature
        logits /= self.T
        mean_logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes) 

        return logits, tmp.long(), sal_loss, inveqv_loss,  mean_logits, mean_labels, spatial_loss


        
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