# this code from: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
from matplotlib.pyplot import fill
import numpy as np
import torch

from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import torchvision.transforms.functional as TF
import math 

fillmask = False
fillcolor = (0, 0, 0)


def ShearX(pair, v, is_feat=0):  # [-0.3, 0.3]
    if is_feat == 0:
        img, mask = pair 
        
        img = TF.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(v)), 0.0],
                resample=Image.BILINEAR,
                fill=fillcolor,
                # center=[0, 0],
            )
        mask = TF.affine(
                mask,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(v)), 0.0],
                resample=Image.NEAREST,
                fill=fillmask,
                # center=[0, 0],
            )
        return img, mask     
    elif is_feat == 1:
        return TF.affine(
            pair,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(v)), 0.0],
            resample=Image.NEAREST,
            fill=fillmask,
            # center=[0, 0],
        )
    else:
        return TF.affine(
            pair,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(v)), 0.0],
            resample=Image.BILINEAR,
            # center=[0, 0],
        )
        

def ShearY(pair, v, is_feat=0):  # [-0.3, 0.3]
    if is_feat == 0:
        img, mask = pair 
        
        img = TF.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(v)),],
                resample=Image.BILINEAR,
                fill=fillcolor,
                # center=[0, 0],
            )
        mask = TF.affine(
                mask,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(v))],
                resample=Image.NEAREST,
                fill=fillmask,
                # center=[0, 0],
            )
        return img, mask     
    elif is_feat == 1:
        return TF.affine(
            pair,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(v)),],
            resample=Image.NEAREST,
            fill=fillmask,
            # center=[0, 0],
        )
    else:
        return TF.affine(
            pair,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(v))],
            resample=Image.BILINEAR,
            # center=[0, 0],
        )
        

def TranslateX(pair, v, is_feat=0):  # [-150, 150] => percentage: [-0.45, 0.45]
    
    if is_feat == 0:
        img, mask = pair 
        v = v * img.size[1]
        img = TF.affine(
                img,
                angle=0.0,
                translate=[v, 0],
                scale=1.0,
                resample=Image.BILINEAR,
                shear=[0.0, 0.0],
                fill=fillcolor,
            )
        mask = TF.affine(
                mask,
                angle=0.0,
                translate=[v, 0],
                scale=1.0,
                resample=Image.NEAREST,
                shear=[0.0, 0.0],
                fill=fillmask,
        )
        return img, mask 

    elif is_feat==1:
        v = v * 224
        return TF.affine(
                pair,
                angle=0.0,
                translate=[v, 0],
                scale=1.0,
                resample=Image.NEAREST,
                shear=[0.0, 0.0],
                fill=fillmask,
            )
    else:
        v = v * 224
        return TF.affine(
                pair,
                angle=0.0,
                translate=[v, 0],
                scale=1.0,
                resample=Image.BILINEAR,
                shear=[0.0, 0.0],
            )
def TranslateY(pair, v, is_feat=0):  # [-150, 150] => percentage: [-0.45, 0.45]
    
    if is_feat == 0:
        img, mask = pair 
        v = v * img.size[0]
        img = TF.affine(
                img,
                angle=0.0,
                translate=[0, v],
                scale=1.0,
                resample=Image.BILINEAR,
                shear=[0.0, 0.0],
                fill=fillcolor,
            )
        mask = TF.affine(
                mask,
                angle=0.0,
                translate=[0, v],
                scale=1.0,
                resample=Image.NEAREST,
                shear=[0.0, 0.0],
                fill=fillmask,
        )
        return img, mask 

    elif is_feat==1:
        
    
        v = v * 224
        return TF.affine(
                pair,
                angle=0.0,
                translate=[0, v],
                scale=1.0,
                resample=Image.NEAREST,
                shear=[0.0, 0.0],
                fill=fillmask,
            )
    else:
        v = v * 224
        return TF.affine(
                pair,
                angle=0.0,
                translate=[0, v],
                scale=1.0,
                resample=Image.BILINEAR,
                shear=[0.0, 0.0],
            )


def Rotate(pair, v, is_feat=0):  # [-30, 30]
    if is_feat == 0:
        img, mask = pair
        img = TF.rotate(img, angle=v, resample=Image.BILINEAR, fill=fillcolor)
        mask = TF.rotate(mask, angle=v, resample=Image.NEAREST, fill=fillmask)
    elif is_feat == 1:
        return TF.rotate(pair, angle=v, resample=Image.NEAREST, fill=fillmask)
    else:
        return TF.rotate(pair, angle=v, resample=Image.BILINEAR)
    return img, mask


def AutoContrast(pair, v, is_feat=0):
    img, mask = pair
    return ImageOps.autocontrast(img), mask


def Invert(pair, v, is_feat=0):
    img, mask = pair
    return ImageOps.invert(img), mask


def Equalize(pair, v, is_feat=0):
    img, mask = pair
    return ImageOps.equalize(img), mask


def Flip(pair, v, is_feat=0):  # not from the paper
    img, mask = pair
    return ImageOps.mirror(img), ImageOps.mirror(mask)


def Solarize(pair, v, is_feat=0):  # [0, 256]
    img, mask = pair
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), mask


def Posterize(pair, v, is_feat=0):  # [4, 8]
    img, mask = pair
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Posterize2(pair, v, is_feat=0):  # [0, 4]
    img, mask = pair
    assert 0 <= v <= 4
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Contrast(pair, v, is_feat=0):  # [0.1,1.9]
    img, mask = pair
    v = v+1
    return ImageEnhance.Contrast(img).enhance(v), mask


def Color(pair, v, is_feat=0):  # [0.1,1.9]
    img, mask = pair
    v = v+1
    return ImageEnhance.Color(img).enhance(v), mask


def Brightness(pair, v, is_feat=0):  # [0.1,1.9]
    img, mask = pair
    v = v+1
    return ImageEnhance.Brightness(img).enhance(v), mask


def Sharpness(pair, v, is_feat=0):  # [0.1,1.9]
    img, mask = pair
    v = v+1
    return ImageEnhance.Sharpness(img).enhance(v), mask



def Identity(pair, v, is_feat=0):
    return pair


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.0, True, False),
        (ShearX, 0., 0.3, True, True),  # 0
        (ShearY, 0., 0.3, True, True),  # 1
        (TranslateX, 0., 0.33, True, True),  # 2
        (TranslateY, 0., 0.33, True, True),  # 3
        (Rotate, 0, 30, True, True),  # 4
        (AutoContrast, 0, 1, False, False),  # 5
        (Invert, 0, 1, False, False),  # 6
        (Equalize, 0, 1, False, False),  # 7
        (Solarize, 0, 110, False, False),  # 8
        (Posterize, 4, 8, False, False),  # 9
        (Color, 0, 0.9, False, True),  # 11
        (Brightness, 0, 0.9, False, True),  # 12
        (Sharpness, 0, 0.9, False, True),  # 13
    ]
    return l



class RandAugment(object):
    def __init__(self, N):
        self.N = N
        
        self.augment_list = augment_list() 
        self.total_ops = len(self.augment_list)
        
        self.magnitude = 30
        
        self.rand_ops = np.array([[random.random() for _ in range(self.total_ops)] for i in range(self.N)])
        self.m = np.array([random.randint(0, self.magnitude) for _ in range(self.N)])

    def __call__(self, index, sample):
        pair = sample['image'], sample['sal']
        
        m = self.m[index]

        for iop, (op, minval, maxval, _, signed) in enumerate(self.augment_list):
            p_seed = self.rand_ops[index][iop]
            if p_seed >= 0.5:
                val = (float(m) / self.magnitude) * float(maxval - minval) + minval
                if p_seed >= 0.75 and signed:
                    val *= -1.0
                
                pair = op(pair, val)
        
        sample['image'] = pair[0]
        sample['sal'] = pair[1]
        
        return sample

    def apply(self, index, feat, is_feat=2):
        
        feat_t = []
        for k, i in enumerate(index):
            m = self.m[i]
            x = feat[k].unsqueeze(0)
            # print(x.shape)

            for iop, (op, minval, maxval, geo, signed) in enumerate(self.augment_list):
                
                if not geo:
                    continue
                
                p_seed = self.rand_ops[i][iop]
                if p_seed >= 0.5:
                    val = (float(m)/ self.magnitude) * float(maxval - minval) + minval 
                    if p_seed >= 0.75 and signed:
                        val *= -1.0
                    x = op(x, val, is_feat=is_feat)
                
            feat_t.append(x)
        feat_t = torch.stack(feat_t)
        return feat_t


        
