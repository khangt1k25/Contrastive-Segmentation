#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from torch import nn
from torch.nn import functional as F
import torch


"""
    SimpleSegmentationModel
    A simple encoder-decoder based segmentation model. 
"""
class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
class ContrastiveSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, head, upsample, use_classification_head=False, use_attention_head=False):
        super(ContrastiveSegmentationModel, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head
        self.use_attention_head = use_attention_head


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
        
        if self.use_attention_head:
            self.attention_head = nn.Conv2d(self.head.in_channels, 1, 1, bias=False)
        

        


    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)
        if self.use_attention:
            mask = self.attention_head(embedding)
            
        
        
        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_attention_head:
                mask = F.interpolate(mask, size=input_shape, mode='bilinear',  align_corners=False)

        # Post Processing and compute [Mask Attention & X attention]
        bsz, dim, h, w = mask.shape
        mask = mask.squeeze(1)
        mask = torch.reshape(mask, shape=(mask.shape[0], -1))
        mask_flat = torch.softmax(mask, dim=1)  
        mask = torch.reshape(mask_flat, shape=(mask.shape[0], h, w))
        x_flat = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))
        mask_flat = mask_flat.unsqueeze(1)
        x_m = (x_flat * mask_flat).sum(-1)


        # Return outputs
        
        if self.use_classification_head and self.use_attention_head:
            return x, sal.squeeze(), x_m, mask 
        elif self.use_classification_head and not self.use_classification_head:
            return x, sal.squeeze()
        elif not self.use_classification_head and self.use_classification_head:
            return x, x_m, mask
        else:
            return x

# class ContrastiveSegmentationModel(nn.Module):
#     def __init__(self, backbone, decoder, head, upsample, use_classification_head=False, freeze_batchnorm='none'):
#         super(ContrastiveSegmentationModel, self).__init__()
#         self.backbone = backbone
#         self.upsample = upsample
#         self.use_classification_head = use_classification_head
        
#         if head == 'linear': 
#             # Head is linear.
#             # We can just use regular decoder since final conv is 1 x 1.
#             self.head = decoder[-1]
#             decoder[-1] = nn.Identity()
#             self.decoder = decoder

#         else:
#             raise NotImplementedError('Head {} is currently not supported'.format(head))


#         if self.use_classification_head: # Add classification head for saliency prediction
#             self.classification_head = nn.Conv2d(self.head.in_channels, 1, 1, bias=False)

#     def forward(self, x):
#         # Standard model
#         input_shape = x.shape[-2:]
#         x = self.backbone(x)
#         embedding = self.decoder(x)

#         # Head
#         x = self.head(embedding)
#         if self.use_classification_head:
#             sal = self.classification_head(embedding)

#         # Upsample to input resolution
#         if self.upsample: 
#             x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
#             if self.use_classification_head:
#                 sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)

#         # Return outputs
#         if self.use_classification_head:
#             return x, sal.squeeze()
#         else:
#             return x

