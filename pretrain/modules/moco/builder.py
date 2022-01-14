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

from utils.common_config import get_model, get_filter
from modules.losses import BalancedCrossEntropyLoss, Regression_loss

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
        
        if p['kernel_size']:
            self.kernel = True
            self.filter = get_filter(p)
        else:
            self.kernel = False

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.dim = p['model_kwargs']['ndim']
        
        self.register_buffer("obj_queue", torch.randn(self.dim, self.K))
        self.obj_queue = nn.functional.normalize(self.obj_queue, dim=0)
        
        self.register_buffer("bg_queue", torch.randn(self.dim, self.K))
        self.bg_queue = nn.functional.normalize(self.bg_queue, dim=0)


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
    def _dequeue_and_enqueue(self, keys_obj, keys_bg):

        batch_size = keys_obj.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.obj_queue[:, ptr:ptr + batch_size] = keys_obj.T
        self.bg_queue[:, ptr:ptr + batch_size] = keys_bg.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr




    def forward(self, im_q, im_k, sal_q, sal_k):
        """
        Input:
            images: a batch of images (B x 3 x H x W) 
            sal: a batch of saliency masks (B x H x W)
        Output:
            logits, targets
        """
        batch_size = im_q.size(0)

        q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        q_reshape = q.reshape(batch_size, self.dim, -1) # B x dim x H.W
        q = q.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # queries: pixels x dim
        

        sal_q_filter = self.filter(sal_q) * (1.-sal_q)
        sal_q_filter = sal_q_filter.reshape(batch_size, -1, 1).type(q.dtype) # B x H.W x 1
        sal_q_flat = sal_q.reshape(batch_size, -1, 1).type(q.dtype)
        

        q_obj_mean = torch.bmm(q_reshape, sal_q_flat).squeeze() # B x dim
        q_obj_mean = nn.functional.normalize(q_obj_mean, dim=1)

        q_bg_mean = torch.bmm(q_reshape, sal_q_filter).squeeze() # B x dim
        q_bg_mean = nn.functional.normalize(q_bg_mean, dim=1)

        q_img_mean = q_obj_mean + q_bg_mean # B x dim
        
        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)
   
        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            sal_q = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            sal_q = sal_q.view(-1)
            mask_indexes = torch.nonzero((sal_q)).view(-1).squeeze()
            sal_q = torch.index_select(sal_q, index=mask_indexes, dim=0) // 2
        
        # compute key prototypes
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            k, _ = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            

            sal_k_filter = self.filter(sal_k) * (1.-sal_k)
            sal_k_filter = sal_k_filter.reshape(batch_size, -1, 1).type(q.dtype) # B x H.W x 1
            sal_k = sal_k.reshape(batch_size, -1, 1).type(q.dtype)
            
            prototypes_obj = torch.bmm(k, sal_k).squeeze() # B x dim
            prototypes_obj = nn.functional.normalize(prototypes_obj, dim=1)

            prototypes_bg = torch.bmm(k, sal_k_filter).squeeze() # B x dim
            prototypes_bg = nn.functional.normalize(prototypes_bg, dim=1)

            prototypes_images = prototypes_obj + prototypes_bg
        
        banks_obj = self.obj_queue.clone().detach()     # shape: dim x negatives
        banks_bg = self.bg_queue.clone().detach()
        banks_image = banks_obj + banks_bg

        # Main from MaskContrast
        # q, k: pixels x dim
        q = torch.index_select(q, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q, prototypes_obj.t())
        l_mem = torch.matmul(q, banks_obj)              # shape: pixels x negatives (Memory bank)
        logits = torch.cat([l_batch, l_mem], dim=1)         # pixels x (proto + negatives)

        
        ## Superpixel 
        l_positive = torch.matmul(q_obj_mean, prototypes_obj.t())
        l_negative = torch.matmul(q_obj_mean, banks_obj)
        l_vs_bg = torch.matmul(q_obj_mean, prototypes_bg.t())
        obj_logits = torch.cat([l_positive, l_negative, l_vs_bg], dim=1)
        obj_labels = torch.arange(obj_logits.shape[0]).to(q.device)


        ## Backgrounds
        bg_positive = torch.matmul(q_bg_mean, prototypes_bg.t())
        bg_negative = torch.matmul(q_bg_mean, banks_bg)
        l_vs_obj = torch.matmul(q_bg_mean, prototypes_obj.t())

        bg_logits = torch.cat([bg_positive, bg_negative, l_vs_obj], dim=1)
        bg_labels = torch.arange(bg_logits.shape[0]).to(q.device)

        ## Images
        img_positive = torch.matmul(q_img_mean, prototypes_images.t())
        img_negative = torch.matmul(q_img_mean, banks_image)
        img_logits = torch.cat([img_positive, img_negative], dim=1)
        img_labels = torch.arange(img_logits.shape[0]).to(q.device)

        # apply temperature
        logits /= self.T
        obj_logits /= self.T
        bg_logits /= self.T
        img_logits /= self.T
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes_obj, prototypes_bg) 

        return logits, sal_q, obj_logits, obj_labels, bg_logits, bg_labels, img_logits, img_labels, sal_loss
