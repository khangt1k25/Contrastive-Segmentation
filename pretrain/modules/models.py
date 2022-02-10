# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module


"""
    ContrastiveSegmentationModel
"""
class ContrastiveSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, head, upsample, use_classification_head=False):
        super(ContrastiveSegmentationModel, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head
        
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

        self.use_recon_head = True
        hidden_dim = 128
        if self.use_recon_head:
            self.recon_head = nn.Sequential(
                nn.Conv2d(self.head.in_channels, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 3, 1),
                nn.Tanh(),
            )


    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)
        if self.use_recon_head:
            recon = self.recon_head(embedding)


        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_recon_head:
                recon = F.interpolate(recon, size=input_shape, mode='bilinear', align_corners=False)
        
        # Return outputs
        if self.use_classification_head and self.use_recon_head:
            return x, recon, sal.squeeze()
        elif self.use_classification_head:
            return x, sal.squeeze()
        else:
            return x

class Filter(nn.Module):
    def __init__(self, kernel_size=3):
  
        super(Filter, self).__init__()
        
        padd = (kernel_size-1)//2

        self.filter = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padd)
        )

    def forward(self, x):
        output = self.filter(x.float())
        
        return output


class PredictionHead(nn.Module):
    def __init__(self, dim=32):
      
        super(PredictionHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
    def forward(self, x):
        output = self.net(x)
        return output