#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import errno
import faiss 
import numpy as np
import torch
import torch.nn as nn
import random

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def freeze_layers(model):
    # Freeze block 1+2 layers in the backbone
    model.model_q.backbone.conv1.eval()
    model.model_q.backbone.bn1.eval()
    model.model_k.backbone.conv1.eval()
    model.model_k.backbone.bn1.eval()
    model.model_q.backbone.layer1.eval()
    model.model_k.backbone.layer1.eval()
    model.model_q.backbone.layer2.eval()
    model.model_k.backbone.layer2.eval()
    for name, param in model.model_q.backbone.conv1.named_parameters():
        param.requires_grad = False
    for name, param in model.model_q.backbone.bn1.named_parameters():
        param.requires_grad = False
    for name, param in model.model_k.backbone.conv1.named_parameters():
        param.requires_grad = False
    for name, param in model.model_k.backbone.bn1.named_parameters():
        param.requires_grad = False
    for name, param in model.model_q.backbone.layer1.named_parameters():
        param.requires_grad = False
    for name, param in model.model_q.backbone.layer2.named_parameters():
        param.requires_grad = False
    for name, param in model.model_k.backbone.layer1.named_parameters():
        param.requires_grad = False
    for name, param in model.model_k.backbone.layer2.named_parameters():
        param.requires_grad = False
    return model




def get_faiss_module(p):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True
    cfg.device     = 0 #NOTE: Single GPU only. 
    idx = faiss.GpuIndexFlatL2(res, p['model_kwargs']['ndim'], cfg)
    return idx

def get_init_centroids(p, K, featlist, index):
    clus = faiss.Clustering(p['model_kwargs']['ndim'], K)
    clus.seed  = np.random.randint(p['seed'])
    clus.niter = p['kmeans']['n_iter']
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)
    return faiss.vector_float_to_array(clus.centroids).reshape(K, p['model_kwargs']['ndim'])

def module_update_centroids(index, centroids):
    index.reset()
    index.add(centroids)
    return index 

def fix_seed_for_reproducability(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic. 

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068 
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi 
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)

def postprocess_label(p, K, idx, idx_img, scores, view):
    
    out = scores[idx].topk(1, dim=0)[1].flatten().detach().cpu().numpy()

    # Save labels.
    if not os.path.exists(os.path.join(p['output_dir'], 'label_' + str(view))):
        os.makedirs(os.path.join(p['output_dir'], 'label_' + str(view)))
    torch.save(out, os.path.join(p['output_dir'], 'label_' + str(view), '{}.pkl'.format(idx_img)))
    
    # Count for re-weighting. 
    counts = torch.tensor(np.bincount(out, minlength=K)).float()
    
    return counts   
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class) # Exclude unlabelled data.
    hist = np.bincount(n_class * label_true[mask] + label_pred[mask],\
                       minlength=n_class ** 2).reshape(n_class, n_class)
    
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        # hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist[lt][lp] += 1 
    return hist


def get_result_metrics(histogram):
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp 

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn) 
    opc = np.sum(tp) / np.sum(histogram)
    
    result = {"iou": iou,
             "mean_iou": np.nanmean(iou),
             "precision_per_class (per class accuracy)": prc,
             "mean_precision (class-avg accuracy)": np.nanmean(prc),
             "overall_precision (pixel accuracy)": opc}

    result = {k: 100*v for k, v in result.items()}

    return result

def compute_negative_euclidean(featmap,  centroids, metric_function):
    centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    return - (1 - 2*metric_function(featmap)+ (centroids*centroids).sum(dim=1).unsqueeze(0))
    

def get_metric_as_conv(centroids):
    C, dim = centroids.size()
    centroids_weight = centroids.unsqueeze(-1).unsqueeze(-1)
    metric_function  = nn.Conv2d(dim, C, 1, padding=0, stride=1, bias=False)
    metric_function.weight.data = centroids_weight
    return metric_function

################################################################################
#                                General torch ops                             #
################################################################################

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False 


def initialize_classifier(p, split='train'):
    if split == 'train':
      classifier = get_linear(p['model_kwargs']['ndim'], p['kmeans']['K_train'])
    else:
      classifier = get_linear(p['model_kwargs']['ndim'], p['kmeans']['K_test'])
    
    return classifier

def get_linear(indim, outdim):
    classifier = nn.Conv2d(indim, outdim, kernel_size=1, stride=1, padding=0, bias=True)
    classifier.weight.data.normal_(0, 0.01)
    classifier.bias.data.zero_()
    return classifier