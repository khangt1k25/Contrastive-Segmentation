# this code from: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import numpy as np
import torch

from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import torchvision.transforms.functional as TF

fillmask = False
fillcolor = (0, 0, 0)


def affine_transform(pair, affine_params, is_feat=False):
    if is_feat:
        return pair.transform(pair.size, Image.AFFINE, affine_params,
                            resample=Image.NEAREST, fillcolor=fillmask)
        # return TF.affine(pair, *affine_params, resample=Image.NEAREST, fillcolor=fillmask)
    else:
        img, mask = pair
        img = img.transform(img.size, Image.AFFINE, affine_params,
                            resample=Image.BILINEAR, fillcolor=fillcolor)
        mask = mask.transform(mask.size, Image.AFFINE, affine_params,
                            resample=Image.NEAREST, fillcolor=fillmask)
        return img, mask


def ShearX(pair, v, seed, is_feat=False):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if seed >= 0.75:
        v = -v
    return affine_transform(pair, (1, v, 0, 0, 1, 0), is_feat=is_feat)


def ShearY(pair, v, seed, is_feat=False):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if seed >= 0.75:
        v = -v
    
    return affine_transform(pair, (1, 0, 0, v, 1, 0), is_feat=is_feat)


def TranslateX(pair, v, seed, is_feat=False):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if seed >= 0.75:
        v = -v
    if is_feat:
        v = v * pair.size[0]
        return affine_transform(pair, (1, 0, v, 0, 1, 0), is_feat=True)
    else:

        img, _ = pair
        v = v * img.size[0]
        return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateY(pair, v, seed, is_feat=False):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if seed >= 0.75:
        v = -v
    if is_feat:
        v = v * pair.size[1]
        return affine_transform(pair, (1, 0, 0, 0, 1, v), is_feat=True)
    else:
        img, _ = pair
        v = v * img.size[1]
        return affine_transform(pair, (1, 0, 0, 0, 1, v))


def TranslateXAbs(pair, v, seed):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if seed >= 0.75:
        v = -v
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateYAbs(pair, v, seed):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if seed >= 0.75:
        v = -v
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def Rotate(pair, v, seed, is_feat=False):  # [-30, 30]
    assert -30 <= v <= 30
    if seed >= 0.75:
        v = -v
    if is_feat:
        pair = pair.rotate(v, resample=Image.NEAREST, fillcolor=fillmask)
        return pair
    else:
        img, mask = pair
        img = img.rotate(v, fillcolor=fillcolor)
        mask = mask.rotate(v, resample=Image.NEAREST, fillcolor=fillmask)
        return img, mask


def AutoContrast(pair, _, seed):
    img, mask = pair
    return ImageOps.autocontrast(img), mask


def Invert(pair, _, seed):
    img, mask = pair
    return ImageOps.invert(img), mask


def Equalize(pair, _, seed):
    img, mask = pair
    return ImageOps.equalize(img), mask


def Flip(pair, _, seed):  # not from the paper
    img, mask = pair
    return ImageOps.mirror(img), ImageOps.mirror(mask)


def Solarize(pair, v, seed):  # [0, 256]
    img, mask = pair
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), mask


def Posterize(pair, v, seed):  # [4, 8]
    img, mask = pair
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Posterize2(pair, v, seed):  # [0, 4]
    img, mask = pair
    assert 0 <= v <= 4
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Contrast(pair, v, seed):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v), mask


def Color(pair, v, seed):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v), mask


def Brightness(pair, v, seed):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v), mask


def Sharpness(pair, v, seed):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v), mask


def Cutout(pair, v, seed):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return pair
    img, mask = pair
    v = v * img.size[0]
    return CutoutAbs(img, v), mask


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Identity(pair, v, seed, is_feat=False):
    return pair


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.0, True),
        (ShearX, 0., 0.3, True),  # 0
        (ShearY, 0., 0.3, True),  # 1
        (TranslateX, 0., 0.33, True),  # 2
        (TranslateY, 0., 0.33, True),  # 3
        (Rotate, 0, 30, True),  # 4
        (AutoContrast, 0, 1, False),  # 5
        (Invert, 0, 1, False),  # 6
        (Equalize, 0, 1, False),  # 7
        (Solarize, 0, 110, False),  # 8
        (Posterize, 4, 8, False),  # 9
        # (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9, False),  # 11
        (Brightness, 0.1, 1.9, False),  # 12
        (Sharpness, 0.1, 1.9, False),  # 13
        # (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
        # (Flip, 1, 1),
    ]
    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment(object):
    def __init__(self, N):
        self.N = N
        
        self.augment_list = augment_list() 
        self.total_ops = len(self.augment_list)
        
        self.magnitude = 30
        
        self.rand_ops = [[random.random() for _ in range(self.total_ops)] for i in range(self.N)]
        self.m = [random.randint(0, self.magnitude) for _ in range(self.N)]

    def __call__(self, index, sample):
        pair = sample['image'], sample['sal']
        
        m = self.m[index]

        for iop, (op, minval, maxval, _) in enumerate(self.augment_list):
            p_seed = self.rand_ops[index][iop]
            if p_seed >= 0.5:
                val = (float(m) / self.magnitude) * float(maxval - minval) + minval
                pair = op(pair, val, p_seed)
        
        sample['image'] = pair[0]
        sample['sal'] = pair[1]
        
        return sample

    def apply(self, index, feat):
        
        feat = TF.to_pil_image(feat)

        m = self.m[index]

        for iop, (op, minval, maxval, geo) in enumerate(self.augment_list):
            p_seed = self.rand_ops[index][iop]
            if geo and p_seed >= 0.5:
                val = (float(m) / self.magnitude) * float(maxval - minval) + minval
                feat = op(feat, val, p_seed, is_feat=True) 

        feat = TF.to_tensor(feat)
        
        return feat   