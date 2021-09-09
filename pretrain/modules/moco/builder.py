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

from utils.common_config import get_model, get_next_transformations
from modules.losses import BalancedCrossEntropyLoss, CatInstConsistency, CatInstContrast, ConsistencyLoss

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

        # for cluster
        self.cons_y = CatInstConsistency(reduction="mean", cons_type="neg_log_dot_prod")
        self.C = p['cluster_kwargs']['C']
        self.smooth_prob = p['cluster_kwargs']['smooth_prob']
        self.smooth_coeff = p['cluster_kwargs']['smooth_coeff']


        # for augment consistency 
        self.transforms = get_next_transformations()
        self.consistency = ConsistencyLoss(type=p['consistency_kwargs']['type'])
        
        # for local contrastive loss
        self.kernel = p['local_contrastive_kwargs']['kernel']
        self.num_local_negatives = p['local_contrastive_kwargs']['num_negatives']


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

    def forward(self, im_q, im_k, sal_q, sal_k, state_dict, transform):
        """
        Input:
            images: a batch of images (B x 3 x H x W) 
            sal: a batch of saliency masks (B x H x W)
        Output:
            logits, targets, local_logits, local_targets
        """
        
        batch_size = im_q.size(0)
        q, bg_q, y_q = self.model_q(im_q)                      # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        flat_q = q.permute((0, 2, 3, 1))                  
        flat_q = torch.reshape(flat_q, [-1, self.dim])    # queries: pixels x dim
        
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
            tmp_for_cluster = tmp.long()
   

        '''
        Prepare prototypes in key size
        '''
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, bg_k, y_k = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)       
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            

             
            # prototypes k
            k_flat = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_k_flat = sal_k.reshape(batch_size, -1, 1).type(k.dtype) # B x H.W x 1
            prototypes_foreground = torch.bmm(k_flat, sal_k_flat).squeeze() # B x dim
            prototypes = nn.functional.normalize(prototypes_foreground, dim=1)        
            
            #prototypes cluster
            if self.p['loss_coeff']['cluster'] > 0:
                ly_2 = y_k
                y_k = torch.softmax(y_k, dim=1)
                y_k = y_k.reshape(batch_size, self.C, -1).type(k.dtype)
                prototypes_cluster = torch.bmm(y_k, sal_k_flat).squeeze()

                prototypes_cluster = nn.functional.normalize(prototypes_cluster, dim=1, p=1.0) # softmax = 1 
     
                prototypes_cluster = torch.index_select(prototypes_cluster, dim=0, index=tmp_for_cluster)


        '''
        Compute Consistency loss
        '''
        consistency_loss = 0 
        if self.p['loss_coeff']['consistency'] > 0:
            if self.p['kornia_version'] == 1:
                inverse_k = []
                inverse_sal = []
                for i in range(len(transform)):

                    sample = {"image": deepcopy(k[i]), 'sal': deepcopy(bg_k[i])}
                    new_sample = self.transforms.inverse(sample, transform[i])
                    inverse_k.append(new_sample['image'].squeeze(0))
                    inverse_sal.append(new_sample['sal'].squeeze(0))

                inverse_k = torch.stack(inverse_k, dim=0).squeeze(0)
                inverse_k = inverse_k.permute((0, 2, 3, 1))                  
                inverse_sal = torch.stack(inverse_sal, dim=0)

                q_selected = q.permute((0, 2, 3, 1))                

                consistency_loss = self.consistency(inverse_k, q_selected, mask=inverse_sal)
                
            elif self.p['kornia_version'] == 2:
                augmented_k = []
                augmented_sal = []
                for i in range(len(state_dict)):

                    sample = {"image": deepcopy(k[i]), 'sal': deepcopy(bg_k[i])}
                    new_sample = self.transforms.forward_with_params(sample, state_dict[i])
                    augmented_sal.append(new_sample['sal'].squeeze(0))
                    augmented_k.append(new_sample['image'].squeeze(0))

                augmented_k = torch.stack(augmented_k, dim=0).squeeze(0)
                augmented_k = augmented_k.permute((0, 2, 3, 1))                  
                augmented_sal = torch.stack(augmented_sal, dim=0)

                q_selected = q.permute((0, 2, 3, 1))

                consistency_loss = self.consistency(augmented_k, q_selected, mask=augmented_sal)
        
        '''
        Compute cluster loss
        '''
        cluster_loss =  0
        entropy = 0
        clamp = 0
        if self.p['loss_coeff']['cluster'] > 0:
            ly_1 = y_q
            y_q = torch.softmax(y_q, dim=1)
            y_q = y_q.permute(0, 2, 3, 1)
            y_q = torch.reshape(y_q, [-1, self.C])
            y_q_object = torch.index_select(y_q, index=mask_indexes, dim=0)
            
            if self.smooth_prob:
                alpha = self.smooth_coeff
                py_1_smt = (1.0 - alpha) * y_q_object + alpha * (1.0 / self.C)
                py_2_smt = (1.0 - alpha) * prototypes_cluster + alpha * (1.0 / self.C)
            else:
                py_1_smt = torch.clamp(y_q_object, 1e-6, 1e6)
                py_2_smt = torch.clamp(prototypes_cluster, 1e-6, 1e6)

            py_avg_1 = y_q_object.mean(0)
            py_avg_2 = prototypes_cluster.mean(0)


            entropy = -0.5 * ((py_avg_1 * py_avg_1.log()).sum(0) +
                                        (py_avg_2 * py_avg_2.log()).sum(0))
            
            #y_batch = torch.matmul(py_1_smt, py_2_smt.t()).log()
            
            #cluster_loss = F.cross_entropy(y_batch, tmp, reduce='mean')
            


            cluster_loss = self.cons_y(py_1_smt, py_2_smt).mean(0)



            zero = torch.zeros([], dtype=torch.float32,
                               device=q.device, requires_grad=False)
            min_rate=0.7
            max_rate=1.3
            lower_clamp_coeff_1 = torch.min(py_avg_1.data - min_rate / self.C, zero).detach()
            lower_clamp_coeff_2 = torch.min(py_avg_2.data - min_rate / self.C, zero).detach()
            lower_clamp = 0.5 * ((py_avg_1 * lower_clamp_coeff_1).sum(0) +
                                  (py_avg_2 * lower_clamp_coeff_2).sum(0))
            tracked_lower_clamp = 0.5 * (lower_clamp_coeff_1.pow(2).sum(0) +
                                          lower_clamp_coeff_2.pow(2).sum(0))


            upper_clamp_coeff_1 = torch.max(py_avg_1.data - max_rate / self.C, zero).detach()
            upper_clamp_coeff_2 = torch.max(py_avg_2.data - max_rate / self.C, zero).detach()
            upper_clamp = 0.5 * ((py_avg_1 * upper_clamp_coeff_1).sum(0) +
                                  (py_avg_2 * upper_clamp_coeff_2).sum(0))
            tracked_upper_clamp = 0.5 * (upper_clamp_coeff_1.pow(2).sum(0) +
                                          upper_clamp_coeff_2.pow(2).sum(0))
            
            
            

            max_abs_logit = 25
            if max_abs_logit > 0:
                lower_logit_clamp_coeff = torch.min(ly_1.data + max_abs_logit, zero).detach()
                lower_logit_clamp_coeff_b = torch.min(ly_2.data + max_abs_logit, zero).detach()
                lower_logit_clamp = 0.5 * ((ly_1 * lower_logit_clamp_coeff).sum(1).mean() +
                                        (ly_2 * lower_logit_clamp_coeff_b).sum(1).mean())
                tracked_lower_logit_clamp = 0.5 * (lower_logit_clamp_coeff.pow(2).sum(1).mean() +
                                                lower_logit_clamp_coeff_b.pow(2).sum(1).mean())

                upper_logit_clamp_coeff = torch.max(ly_1.data - max_abs_logit, zero).detach()
                upper_logit_clamp_coeff_b = torch.max(ly_2.data - max_abs_logit, zero).detach()
                upper_logit_clamp = 0.5 * ((ly_1 * upper_logit_clamp_coeff).sum(1).mean() +
                                        (ly_2 * upper_logit_clamp_coeff_b).sum(1).mean())

                tracked_upper_logit_clamp = 0.5 * (upper_logit_clamp_coeff.pow(2).sum(1).mean() +
                                                upper_logit_clamp_coeff_b.pow(2).sum(1).mean())

            else:
                lower_logit_clamp = zero
                tracked_lower_logit_clamp = zero

                upper_logit_clamp = zero
                tracked_upper_logit_clamp = zero
            
            clamp = 0.01 * upper_clamp + 0.01 * lower_clamp + 0.01*lower_logit_clamp+0.01*upper_logit_clamp
        ''' 
        Compute local contrastive logits, labels 
        '''
        l_logits = []
        l_labels = []
        if self.p['loss_coeff']['local_contrastive'] > 0:
            l_logits = []
            for i in range(q.shape[0]):
                # Working with each image
                indexes = torch.nonzero((sal_q[i]).view(-1)).squeeze()      # indexes: opixels
                q_i = q[i].view(-1, self.dim)                               # q_i:  HW x dim
                object_i = q_i[indexes]                                     # object_i: opixels x dim
                
                local_i = F.avg_pool2d(q[i], kernel_size=self.kernel, stride=1, padding=1)
                local_i = local_i.view(-1, self.dim)[indexes]


                local_positive  = torch.einsum('ij,ji->i', object_i, local_i.T)  # [opixels]


                neg_indexes = torch.randint(low=0, high=q_i.shape[0], size=(indexes.shape[0], self.num_local_negatives))
                neg = q_i[neg_indexes]
                local_negative = torch.bmm(neg, object_i.unsqueeze(-1)).squeeze(-1)
                

                local_logits = torch.cat([local_positive.view(-1, 1), local_negative], dim=1)

                l_logits.append(local_logits)

            l_logits = torch.cat(l_logits, dim=0)
            l_labels = torch.zeros(l_logits.shape[0]).to(sal_q.device)
            l_labels = l_labels.long()
            l_logits /= self.T

        


        '''
        Compute Object Contrastive loss 
        '''
        q = torch.index_select(flat_q, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q, prototypes.t())   # shape: pixels x proto
        negatives = self.queue.clone().detach()          # shape: dim x negatives
        l_mem = torch.matmul(q, negatives)          # shape: pixels x negatives (Memory bank)
        logits = torch.cat([l_batch, l_mem], dim=1)      # pixels x (proto + negatives)

        # apply temperature
        logits /= self.T
        

        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes) 

        return logits, tmp, l_logits, l_labels, sal_loss, consistency_loss, cluster_loss, entropy, clamp


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



