#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from numpy.core.fromnumeric import shape
import torch
from torch import nn
from torch.nn import functional as F


"""
    ContrastiveSegmentationModel
"""
class ContrastiveSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, head, upsample, use_classification_head=False):
        super(ContrastiveSegmentationModel, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head
        # self.use_y_head = use_y_head
        # self.C = C

        if head == 'linear': 
            # Head is linear.
            # We can just use regular decoder since final conv is 1 x 1.
            self.head = decoder[-1]
            decoder[-1] = nn.Identity()
            self.decoder = decoder

        else:
            raise NotImplementedError('Head {} is currently not supported'.format(head))

        if self.use_classification_head: # Add classification head for saliency prediction
            self.classification_head = nn.Conv2d(self.head.in_channels, 1, 1, bias=False)
        # if self.use_y_head:
        #     self.y_head = nn.Conv2d(self.head.in_channels, self.C, 1)

        


    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)
        # if self.use_y_head:
        #     y = self.y_head(embedding)
        
        
        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
            # if self.use_y_head:
            #     y = F.interpolate(y, size=input_shape, mode='bilinear',  align_corners=False)
        
        # Return outputs
        
        if self.use_classification_head:
            return x, sal.squeeze()
            
        else:
            return x






class AttentionHead(nn.Module):
    def __init__(self, dim):
        super(AttentionHead, self).__init__()
        
        self.dim = dim

        self.attention = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z, sal_z):
        '''
        Input:
        z: (bsz, dim, H, W)
        sal_z: (bsz, H, W)
        Output:
        zM = (bsz, dim)
        '''
        bsz, dim, h, w = z.shape
        mask = self.attention(z)
        mask = mask.squeeze(1)
        mask = torch.reshape(mask, shape=(mask.shape[0], -1))
        mask_flat = torch.softmax(mask, dim=1)
        
        mask = torch.reshape(mask_flat, shape=(mask.shape[0], h, w))
        
        # mask  = mask * sal_z
        # mask[mask==0.] = float('-inf') 
        # mask = torch.reshape(mask, shape=(mask.shape[0], -1))
        # mask = torch.softmax(mask, dim=1)
        # mask = mask.unsqueeze(1)

        z_flat = torch.reshape(z, shape=(z.shape[0], z.shape[1], -1))
        mask_flat = mask_flat.unsqueeze(1)
        
      
        z_m = (z_flat * mask_flat).sum(-1)


        mask[sal_z==1.] = 1.

        return z_m, mask

    