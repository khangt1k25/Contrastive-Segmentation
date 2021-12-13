#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from numpy.core.fromnumeric import shape
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
import math
import numbers

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

    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)
            
            
        
        
        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
        # Return outputs
        
        if self.use_classification_head:
            return x, sal.squeeze()
        else:
            return x


class PredictionHead(nn.Module):
    def __init__(self, dim):
        self.dim = int(dim) 
        super(PredictionHead, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        output = self.encoder(x)

        output = self.decoder(output)
        
        return output


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

class Filter(nn.Module):
    def __init__(self):
  
        super(Filter, self).__init__()

        self.filter = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1) ## change filter here
        )

    def forward(self, x):
        output = self.filter(x.float())
        
        return output

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        self.pad = F.pad 
        self.pad_value = (1, 1, 1, 1)

    def forward(self, input):
        output = input.unsqueeze(1)

        output = self.pad(output, self.pad_value)

        output  = self.conv(output, weight=self.weight, groups=self.groups)
        
        return output