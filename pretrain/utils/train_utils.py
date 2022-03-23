#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from audioop import cross
from cv2 import reduce
import torch
from torch.nn.functional import cross_entropy
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter, writer 
import os
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm 
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans 
import torch.utils.data as data
import faiss 
from modules.losses import FocalLoss


def run_mini_batch_kmeans(p, dataloader, model, split='train', seed=2022):
    '''
    Clustering for Key view
    '''
    kmeans_loss  = AverageMeter('kmean loss')
    faiss_module = get_faiss_module(p)
    
    if split=='train':
        K = p['kmeans']['K_train']
    elif split == 'test':
        K = p['kmeans']['K_test']
    

    data_count   = np.zeros(K)
    featslist    = []
    num_batches  = 0
    first_batch  = True    
    isreduce = (p['kmeans']['reducer'] > 0)
    
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            
            img_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
            sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)

            k, _ = model.model_k(img_k) # Bx dim x H x W
            k = nn.functional.normalize(k, dim=1)
            batch_size = k.shape[0]
            k = k.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
            k = torch.reshape(k, [-1, p['model_kwargs']['ndim']]) # queries: BHW x dim
            
            # Drop background pixels
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_k.device)
            sal_k = (sal_k + torch.reshape(offset, [-1, 1, 1]))*sal_k 
            sal_k = sal_k.view(-1)
            mask_indexes = torch.nonzero((sal_k)).view(-1).squeeze()
    
            if isreduce:
                reducer_idx = torch.randperm(mask_indexes.shape[0])[:p['kmeans']['reducer']*batch_size]
                mask_indexes = mask_indexes[reducer_idx]
            
            k = torch.index_select(k, index=mask_indexes, dim=0).detach().cpu() 
            

            if i_batch == 0:
                print('Batch feature : {}'.format(list(k.shape)))
            
            if num_batches < p['kmeans']['n_init']:
                featslist.append(k)
                num_batches += 1
                if num_batches == p['kmeans']['n_init'] or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(p, K, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)
                        kmeans_loss.update(D.mean())
                        print('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
                    else:
                        b_feat = torch.cat(featslist)
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat.numpy().astype('float32'), 1)
                        kmeans_loss.update(D.mean())

                        # Update centroids. 
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr    = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k]   = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(0).numpy().astype('float32')
                    
                    # Empty. 
                    featslist   = []
                    num_batches = p['kmeans']['n_init'] - p['kmeans']['n_update']

            if (i_batch % 100) == 0:
                print('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))
    
    centroids = torch.tensor(centroids, requires_grad=False).cuda()
    centroids = F.normalize(centroids, dim=1)
    return centroids, kmeans_loss.avg



def compute_labels(p, logger, dataloader, model, centroids, device):
    """
    Label for Query view using Key view: Eqv
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    K = centroids.size(0) + 1

    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids)
    metric_function = metric_function.to(device)
    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            img_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
            # sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
            indice = batch['query']['meta']['name']

            
            q, _ = model.model_k(img_q) # Bx dim x H x W
            q = nn.functional.normalize(q, dim=1)

            if i_batch == 0:
                print('Centroid size      : {}'.format(list(centroids.shape)))
                print('Batch input size   : {}'.format(list(img_q.shape)))
                print('Batch feature size : {}\n'.format(list(q.shape)))

            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(q, centroids, metric_function) #BxCxHxW: all bg 're 0 
            

            # Save labels and count. 
            for idx, idx_img in enumerate(indice):
                counts += postprocess_label(p, K, idx, idx_img, scores, view='query')
            
            if (i_batch % 200) == 0:
                print('[Assigning labels] {} / {}'.format(i_batch, len(dataloader)))
    
    weight = counts / counts.sum()
     
    return weight




def train(p, train_loader, model, optimizer, epoch):
    losses = AverageMeter('Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive', ':.4e')
    saliency_losses = AverageMeter('CE', ':.4e')
    cluster_losses = AverageMeter('Cluster', ':.4e')
    kmeans_losses = AverageMeter('Kmeans', ':.4e')
    top1contrast = AverageMeter('Acc1@contrastive', ':6.2f')
    top1cluster = AverageMeter('Acc1@cluster', ':6.2f')
    progress = ProgressMeter(len(train_loader), 
                        [losses, contrastive_losses, cluster_losses, kmeans_losses, saliency_losses, top1contrast, top1cluster],
                        prefix="Epoch: [{}]".format(epoch))
    

    if p['freeze_layers']:
        model = freeze_layers(model)
    
    
    if epoch % p['kmeans']['cluster_epochs'] == 0:
        centroids, kmloss = run_mini_batch_kmeans(p, train_loader, model, split='train')
        kmeans_losses.update(kmloss)
        classifier = initialize_classifier(p, split='train')
        classifier = classifier.cuda()
        classifier.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
        freeze_all(classifier)
    else:
        classifier = None

    # weight = compute_labels(p, train_loader, model, centroids) 
    
    model.train()
    for i, batch in enumerate(train_loader):
        # Forward pass
        im_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
        sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
        im_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
        sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
        

        if classifier:
            logits, labels, cluster_logits, cluster_labels, saliency_loss = model(im_q=im_q, sal_q=sal_q, im_k=im_k, sal_k=sal_k, classifier=classifier)
        else:
            logits, labels, saliency_loss = model(im_q=im_q, sal_q=sal_q, im_k=im_k, sal_k=sal_k)


        #Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = cross_entropy(logits, labels, weight=w_class,
                                            reduction='mean')

        
        if classifier:
            
            focal = False
            if focal:
                focal_loss = FocalLoss(gamma=3, reduction='mean')
                cluster_loss  = focal_loss(cluster_logits, cluster_labels)
            else:
                cluster_loss = cross_entropy(cluster_logits, cluster_labels, reduction='mean')
            
            
            cluster_losses.update(cluster_loss.item())
            loss = contrastive_loss + saliency_loss + p['loss_coeff']['cluster'] * cluster_loss
            bcc1, _ = accuracy(cluster_logits, cluster_labels, topk=(1, 5))
            top1cluster.update(bcc1[0], im_q.size(0))
        else:
            loss = contrastive_loss + saliency_loss

        contrastive_losses.update(contrastive_loss.item())
        saliency_losses.update(saliency_loss.item())
        losses.update(loss.item())
        acc1, _ = accuracy(logits, labels, topk=(1, 5))
        top1contrast.update(acc1[0], im_q.size(0))
        

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Display progress
        if i % 25 == 0:
            progress.display(i)
    
    writer_path = os.path.join(p['output_dir'], "runs")
    writer = SummaryWriter(log_dir=writer_path)
    writer.add_scalar('total loss', losses.avg, epoch)
    writer.add_scalar('contrastive loss', contrastive_losses.avg, epoch)
    writer.add_scalar('saliency loss', saliency_losses.avg, epoch)
    writer.add_scalar('cluster loss', cluster_losses.avg, epoch)
    writer.add_scalar('kmeans loss', kmeans_losses.avg, epoch)

    writer.close()      


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    