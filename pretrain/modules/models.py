#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
from torch import nn
from torch.nn import functional as F


"""
    ContrastiveSegmentationModel
"""
class ContrastiveSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, head, upsample, use_classification_head=False, use_y_head=False, C=20):
        super(ContrastiveSegmentationModel, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head
        self.use_y_head = use_y_head
        self.C = C

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
        if self.use_y_head:
            self.y_head = nn.Conv2d(self.head.in_channels, self.C, 1, bias=False)


    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)
        if self.use_y_head:
            y = self.y_head(embedding)



        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_y_head:
                y = F.interpolate(y, size=input_shape, mode='bilinear',  align_corners=False)

        # Return outputs
        if self.use_classification_head and self.use_y_head:
            return x, sal.squeeze(), y
        elif self.use_classification_head:
            return x, sal.squeeze()
        elif self.use_y_head:
            return x, y
        else:
            return x
