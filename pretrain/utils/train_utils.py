#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from audioop import cross
from cv2 import reduce
import torch
from torch.nn.functional import cross_entropy
from utils.utils import AverageMeter, ProgressMeter, freeze_layers
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




def get_faiss_module(in_dim):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device     = 0 #NOTE: Single GPU only. 
    idx = faiss.GpuIndexFlatL2(res, in_dim, cfg)

    return idx


def get_init_centroids(in_dim, seed, K, featlist, index):

    clus = faiss.Clustering(in_dim, K)
    clus.seed  = np.random.randint(seed)
    clus.niter = 30 #fix
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)

    return faiss.vector_float_to_array(clus.centroids).reshape(K, in_dim)

def module_update_centroids(index, centroids):

    index.reset()
    index.add(centroids)

    return index 

def run_mini_batch_kmeans(p, dataloader, model):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update. 
    """

    num_init_batches = 64  
    arg_num_batches  = 64
    reducer = 100 # no pixels per image
    in_dim = 32
    K_train = 20


    kmeans_loss  = AverageMeter("Kmeans loss")
    faiss_module = get_faiss_module(in_dim=in_dim)
    data_count   = np.zeros(K_train)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    
    
    for i_batch, batch in enumerate(dataloader):
        with torch.no_grad():
            img_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
            sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
            
            indices = batch['key']['meta']['index'].long().cpu().numpy()

            k, _ = model.model_k(img_k) # Bx dim x H x W
            k = nn.functional.normalize(k, dim=1)
            batch_size, dim = k.shape[0], k.shape[1]

            k = k.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
            k = torch.reshape(k, [-1, dim]) # queries: BHW x dim

            offset = torch.arange(0, 2 * batch_size, 2).to(sal_k.device)
            sal_k = (sal_k + torch.reshape(offset, [-1, 1, 1]))*sal_k 
            sal_k = sal_k.view(-1)
            
            mask_indexes = torch.nonzero((sal_k)).view(-1).squeeze()
            reducer_idx = torch.randperm(mask_indexes.shape[0])[:reducer*batch_size]
            mask_indexes = mask_indexes[reducer_idx]
            
            k = torch.index_select(k, index=mask_indexes, dim=0).detach().cpu() # pixels x dim 
            
            
            

            if i_batch == 0:
                print('Batch input size : {}'.format(list(img_k.shape)))
                print('Batch feature : {}'.format(list(k.shape)))


            if num_batches < num_init_batches:
                featslist.append(k)
                num_batches += 1
                
                if num_batches == num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')

                        print(featslist.shape)

                        centroids = get_init_centroids(in_dim=32, seed=2022, K=K_train, featlist=featslist, index=faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)
                        
                        kmeans_loss.update(D.mean())
                        
                        print('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        
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
                    num_batches = num_init_batches - arg_num_batches

            if (i_batch % 100) == 0:
                print('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))

    centroids = torch.tensor(centroids, requires_grad=False).cuda()
    
    return centroids, kmeans_loss.avg

def compute_negative_euclidean(featmap, sal,  centroids, metric_function):
    '''
        featmap: BxdimxHxW
        sal: BxHxW
        centroids: Cxdim
        metrics: conv1x1: dim->C
    '''
    predicted = metric_function(featmap)
    predicted = predicted.permute((0, 2, 3, 1)) #BxHxWxC
    predicted = predicted*sal # mask background

     #
    # centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    # return - (1 - 2*metric_function(featmap)\
    #             + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
        

def get_metric_as_conv(centroids):
    C, dim = centroids.size()

    centroids_weight = centroids.unsqueeze(-1).unsqueeze(-1)
    metric_function  = nn.Conv2d(dim, C, 1, padding=0, stride=1, bias=False)
    metric_function.weight.data = centroids_weight  
    metric_function = metric_function.cuda()
    
    return metric_function

def postprocess_label(args, K, idx, idx_img, scores):
    out = scores[idx].topk(1, dim=0)[1].flatten().detach().cpu().numpy()

    # Save labels. 
    if not os.path.exists(os.path.join(args.save_model_path, 'label_' + str(n_dual))):
        os.makedirs(os.path.join(args.save_model_path, 'label_' + str(n_dual)))
    torch.save(out, os.path.join(args.save_model_path, 'label_' + str(n_dual), '{}.pkl'.format(idx_img)))
    
    # Count for re-weighting. 
    counts = torch.tensor(np.bincount(out, minlength=K)).float()

    return counts 
def compute_labels(p, dataloader, model, centroids):
    """
    Label all images for each view with the obtained cluster centroids. 
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    K = centroids.size(0)



    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids)

    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
            sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
            
            indices = batch['key']['meta']['index'].long().cpu().numpy()

            q, _ = model.model_q(img_q) # Bx dim x H x W
            q = nn.functional.normalize(q, dim=1)
            batch_size, dim = q.shape[0], q.shape[1]
 
            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(q, sal_q, centroids, metric_function) 

            # Save labels and count. 
            for idx, idx_img in enumerate(indices):
                counts += postprocess_label(p, K, idx, idx_img, scores)

            if (i % 200) == 0:
                print('[Assigning labels] {} / {}'.format(i, len(dataloader)))
    weight = counts / counts.sum()
        
    return weight

def train(p, N, train_loader, model, optimizer, epoch, amp):
    losses = AverageMeter('Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive', ':.4e')
    saliency_losses = AverageMeter('CE', ':.4e')
  
    kmeans_losses = AverageMeter('Kmeans', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), 
                        [losses, contrastive_losses, kmeans_losses, saliency_losses, top1, top5],
                        prefix="Epoch: [{}]".format(epoch))
    model.train()

    if p['freeze_layers']:
        model = freeze_layers(model)
    
    centroids, kmloss = run_mini_batch_kmeans(p, train_loader, model)
    kmeans_losses.update(kmloss)

    weight = compute_labels(p, train_loader, model, centroids) 



#     for i, batch in enumerate(train_loader):
#         # Forward pass
#         im_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
#         sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
#         im_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
#         sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
        

#         logits, labels, saliency_loss = model(im_q=im_q, sal_q=sal_q, im_k=im_k, sal_k=sal_k, centroids=centroids)


#         # Use E-Net weighting for calculating the pixel-wise loss.
#         # uniq, freq = torch.unique(labels, return_counts=True)
#         # p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
#         # p_class_non_zero_classes = freq.float() / labels.numel()
#         # p_class[uniq] = p_class_non_zero_classes
#         # w_class = 1 / torch.log(1.02 + p_class)
#         # contrastive_loss = cross_entropy(logits, labels, weight=w_class,
#         #                                     reduction='mean')

#         contrastive_loss = cross_entropy(logits, labels,
#                                             reduction='mean')

    

#     #     # Calculate total loss and update meters
#         loss = contrastive_loss + saliency_loss 
        
#         contrastive_losses.update(contrastive_loss.item())
#         saliency_losses.update(saliency_loss.item())


#         losses.update(loss.item())
        
        


#         acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
#         top1.update(acc1[0], im_q.size(0))
#         top5.update(acc5[0], im_q.size(0))
        

#         # Update model
#         optimizer.zero_grad()
#         if amp is not None: # Mixed precision
#             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                 scaled_loss.backward()            
#         else:
#             loss.backward()
#         optimizer.step()

#         # Display progress
#         if i % 25 == 0:
#             progress.display(i)
    
#     writer_path = os.path.join(p['output_dir'], "runs")
#     writer = SummaryWriter(log_dir=writer_path)
#     writer.add_scalar('total loss', losses.avg, epoch)
#     writer.add_scalar('contrastive loss', contrastive_losses.avg, epoch)
#     writer.add_scalar('saliency loss', saliency_losses.avg, epoch)
#     writer.add_scalar('kmeans loss', kmeans_losses.avg, epoch)
#     writer.close()      


# @torch.no_grad()
# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
