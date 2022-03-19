
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import numpy as np
import torch
from PIL import Image
from utils.utils import *
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from termcolor import colored
from joblib import Parallel, delayed
from sklearn import metrics

N_JOBS = 1 # set to number of threads


def eval_kmeans(p, val_dataset, n_clusters=21, compute_metrics=False, verbose=True):
    n_classes = p['num_classes'] + int(p['has_bg'])

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
  
    # Load all pixel embeddings
    all_pixels = np.zeros((len(val_dataset) * 500 * 500), dtype=np.float32)
    all_gt = np.zeros((len(val_dataset) * 500 * 500), dtype=np.float32)
    offset_ = 0

    for i, sample in enumerate(val_dataset):
        if i % 300 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(val_dataset)))

        # Load embedding
        filename = os.path.join(p['embedding_dir'], sample['meta']['image'] + '.npy')
        embedding = np.load(filename)

        # Check where ground-truth is valid. Append valid pixels to the array.
        gt = sample['semseg']
        valid = (gt != 255)
        n_valid = np.sum(valid)
        all_gt[offset_:offset_+n_valid] = gt[valid]

        # print(embedding.shape)
        # Possibly reshape embedding to match gt.
        if embedding.shape != gt.shape:
            embedding = cv2.resize(embedding, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # print(embedding.shape)
        # print(gt.shape)

        # Put the reshaped ground truth in the array
        all_pixels[offset_:offset_+n_valid,] = embedding[valid]
        all_gt[offset_:offset_+n_valid,] = gt[valid]

        # Update offset_
        offset_ += n_valid

    # All pixels, all ground-truth
    all_pixels = all_pixels[:offset_,]
    all_gt = all_gt[:offset_,]

    # Do hungarian matching
    print(colored('Starting hungarian', 'green'))
    num_elems = offset_
    if n_clusters == n_classes:
        print('Using hungarian algorithm for matching')
        match = _hungarian_match(all_pixels, all_gt, preds_k=n_clusters, targets_k=n_classes, metric='iou')

    else:
        print('Using majority voting for matching')
        match = _majority_vote(all_pixels, all_gt, preds_k=n_clusters, targets_k=n_classes)

    # Remap predictions
    reordered_preds = np.zeros(num_elems, dtype=all_pixels.dtype)
    for pred_i, target_i in match:
        reordered_preds[all_pixels == int(pred_i)] = int(target_i)

    if compute_metrics:
        print('Computing acc, nmi, ari ...')
        acc = int((reordered_preds == all_gt).sum()) / float(num_elems)
        nmi = metrics.normalized_mutual_info_score(all_gt, reordered_preds)
        ari = metrics.adjusted_rand_score(all_gt, reordered_preds)
    else: 
        acc, nmi, ari = None, None, None

    # TP, FP, and FN evaluation
    print(colored('Starting miou', 'green'))
    for i_part in range(0, n_classes):
        tmp_all_gt = (all_gt == i_part)
        tmp_pred = (reordered_preds == i_part)
        tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
        fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
        fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
    eval_result['acc'] = acc
    eval_result['nmi'] = nmi
    eval_result['ari'] = ari
        
    if verbose:
        print('Evaluation of semantic segmentation ')
        print('mIoU is %.2f' %(100*eval_result['mIoU']))
        class_names = val_dataset.get_class_names()
        for i_part in range(n_classes):
            print('IoU class %s is %.2f' %(class_names[i_part], 100*jac[i_part]))

    print(eval_result)

    return eval_result


@torch.no_grad()
def save_embeddings_to_disk(p, val_loader, model, n_clusters=21, seed=1234):
    import torch.nn as nn
    print('Save embeddings to disk ...')
    model.eval()
    ptr = 0

    all_prototypes = torch.zeros((len(val_loader.sampler), 32)).cuda()
    all_sals = torch.zeros((len(val_loader.sampler), 512, 512)).cuda()
    names = []
    for i, batch in enumerate(val_loader):
        output, sal = model(batch['image'].cuda(non_blocking=True))
        meta = batch['meta']

        # compute prototypes
        bs, dim, _, _ = output.shape
        output = output.reshape(bs, dim, -1) # B x dim x H.W
        sal_proto = sal.reshape(bs, -1, 1).type(output.dtype) # B x H.W x 1
        prototypes = torch.bmm(output, sal_proto*(sal_proto>0.5).float()).squeeze() # B x dim
        prototypes = nn.functional.normalize(prototypes, dim=1)        
        all_prototypes[ptr: ptr + bs] = prototypes
        all_sals[ptr: ptr + bs, :, :] = (sal > 0.5).float()
        ptr += bs
        for name in meta['image']:
            names.append(name)

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    # perform kmeans
    all_prototypes = all_prototypes.cpu().numpy()
    all_sals = all_sals.cpu().numpy()
    n_clusters = n_clusters - 1
    print('Kmeans clustering to {} clusters'.format(n_clusters))
    
    print(colored('Starting kmeans with scikit', 'green'))
    pca = PCA(n_components = 32, whiten = True)
    all_prototypes = pca.fit_transform(all_prototypes)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    prediction_kmeans = kmeans.fit_predict(all_prototypes)

    # save predictions
    for i, fname, pred in zip(range(len(val_loader.sampler)), names, prediction_kmeans):
        prediction = all_sals[i].copy()
        prediction[prediction == 1] = pred + 1
        np.save(os.path.join(p['embedding_dir'], fname + '.npy'), prediction)
        if i % 300 == 0:
            print('Saving results: {} of {} objects'.format(i, len(val_loader.dataset)))


@torch.no_grad()
def save_embedding_to_disk_pixel(p, val_loader, model, n_clusters=21, seed=1234):
    import torch.nn as nn
    print('Save embeddings to disk ...')
    model.eval()
    isreduce = False
    reduce = 100
    names = []
    n_init = 64
    n_update = 1
    num_batches = 0
    kmeans_loss  = AverageMeter('kmean loss')
    faiss_module = get_faiss_module(p)
    featslist = []
    n_clusters = n_clusters - 1
    data_count   = np.zeros(n_clusters)
    first_batch = True
    
    for i_batch, batch in enumerate(val_loader):
        output, sal = model(batch['image'].cuda(non_blocking=True))
        meta = batch['meta'] 

        

        output = nn.functional.normalize(output, dim=1)
        batch_size, ndim = output.shape[0], output.shape[1]
        output = output.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        output = torch.reshape(output, [-1, ndim]) # BHW x dim
        
        # Drop background pixels
        sal_proto = (sal>0.5).float()
        
        offset = torch.arange(0, 2 * batch_size, 2).to(sal.device)
        sal_proto = (sal_proto + torch.reshape(offset, [-1, 1, 1]))*sal_proto 
        sal_proto = sal_proto.view(-1)
        mask_indexes = torch.nonzero((sal_proto)).view(-1).squeeze()

        if isreduce:
            reducer_idx = torch.randperm(mask_indexes.shape[0])[:reduce*batch_size]
            mask_indexes = mask_indexes[reducer_idx]
        
        output = torch.index_select(output, index=mask_indexes, dim=0).detach().cpu()

        
        if num_batches < n_init:
            featslist.append(output)
            num_batches += 1
            if num_batches == n_init or num_batches == len(val_loader):
                if first_batch:
                    featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                    centroids = get_init_centroids(p, n_clusters, featslist, faiss_module, seed).astype('float32')
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
                num_batches = n_init - n_update

        if (i_batch % 100) == 0:
            print('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(val_loader), kmeans_loss.avg))


    centroids = torch.tensor(centroids, requires_grad=False).cuda() # Cluster x dim
    # centroids = nn.functional.normalize(centroids, dim=1)

    classifier = initialize_classifier()
    classifier = classifier.cuda()
    classifier.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
    freeze_all(classifier)

    
    for i_batch, batch in enumerate(val_loader):
        output, sal = model(batch['image'].cuda(non_blocking=True))
        meta = batch['meta'] 
        sal = (sal > 0.5).float()
        preds = classifier(output) # Bx Cluster x H x W
        preds = preds.topk(1, dim=1)[1].squeeze() # Bx H x W
        preds = ((preds + 1) * sal).long().cpu().numpy() # BxHxW

  

        # for i, fname in enumerate(meta['image']):
        for i in range(preds.shape[0]):
            fname = meta['image'][i]
            np.save(os.path.join(p['embedding_dir'], fname + '.npy'), preds[i])
        
        if i_batch % 100 == 0:
            print('Saving results: {} of {} objects'.format(i_batch, len(val_loader.dataset)))

    

def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k, metric='acc'):
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k

    # perform hungarian matching
    print('Using iou as metric')
    # results = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(delayed(get_iou)(flat_preds, flat_targets, c1, c2) for c2 in range(num_k) for c1 in range(num_k))
    results = [[get_iou(flat_preds, flat_targets, c1, c2) for c2 in range(num_k)] for c1 in range(num_k)]

    results = np.array(results)
    results = results.reshape((num_k, num_k)).T
    match = linear_sum_assignment(flat_targets.shape[0] - results)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def _majority_vote(flat_preds, flat_targets, preds_k, targets_k):
    # iou_mat = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(delayed(get_iou)(flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = (get_iou(flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)
    results = iou_mat.reshape((targets_k, preds_k)).T
    results = np.argmax(results, axis=1)
    match = np.array(list(zip(range(preds_k), results)))
    return match


def get_iou(flat_preds, flat_targets, c1, c2):
    tp = 0
    fn = 0
    fp = 0
    tmp_all_gt = (flat_preds == c1)
    tmp_pred = (flat_targets == c2)
    tp += np.sum(tmp_all_gt & tmp_pred)
    fp += np.sum(~tmp_all_gt & tmp_pred)
    fn += np.sum(tmp_all_gt & ~tmp_pred)
    jac = float(tp) / max(float(tp + fp + fn), 1e-8)
    return jac