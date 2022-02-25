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


def get_anchor_features(all_features, all_assignments):
    
    n_cluster = len(np.unique(all_assignments))
    anchor = np.zeros((n_cluster, all_features.shape[-1]))
    
    for c in range(n_cluster):        
        
        c_features = np.mean(all_features[all_assignments==c], axis = 0)

        
        anchor[c,:] = c_features


    return anchor


class PseudoDataset(data.Dataset):
    
    def __init__(self, indices, assigments):
        super(PseudoDataset, self).__init__()

        self.indices = indices
        self.assigments = assigments
        
        assert(len(self.indices) == len(self.assigments))

        print('Number of samples {}'.format(len(indices)))

    def __getitem__(self, index):
        return self.assigments[index]
        
    def __len__(self):
            return len(self.indices)


    def __str__(self):
        return 'Pseudo dataset'




def get_all_features(p, train_loader, model, N):
    
    for i, batch in tqdm(enumerate(train_loader)):
        # Forward pass
        with torch.no_grad():
            img = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
            sal = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
            indices = batch['key']['meta']['index']
            indices = indices.long().cpu().numpy()
            
            feat, _ = model.model_k(img) 
            bsz = feat.shape[0]
            dim = feat.shape[1]
            feat = nn.functional.normalize(feat, dim=1)
            feat = feat.reshape(bsz, dim, -1) # B x dim x H.W

            sal = sal.reshape(bsz, -1, 1).type(feat.dtype)

            feat_mean = torch.bmm(feat, sal).squeeze() # B x dim
            feat_mean = nn.functional.normalize(feat_mean, dim=1)  # Bx dim
            
            feat_mean = feat_mean.cpu().numpy()

            if i == 0:
                all_feat = np.zeros((N, dim))
                all_indices = np.zeros((N, ))
            # if i==5:
            #   break
            
            all_feat[i * bsz: (i + 1) * bsz] = feat_mean
            all_indices[i * bsz: (i + 1) * bsz] = indices
    
    # all_feat = all_feat[:(i+1)*bsz]
    # all_indices = all_indices[:(i+1)*bsz]
      
    return all_feat, all_indices

def cluster_all_features(all_features, n_clusters=20, seed=2022):
    print('Start PCA.')
    pca = PCA(n_components = 32, whiten = True)
    all_features = pca.fit_transform(all_features)
    print('Start Kmeans clustering to {} clusters'.format(n_clusters))
    # # kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    prediction_kmeans = kmeans.fit_predict(all_features)
    
    return prediction_kmeans

def get_faiss_module(in_dim):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device     = 0 #NOTE: Single GPU only. 
    idx = faiss.GpuIndexFlatL2(res, in_dim, cfg)

    return idx

def feature_flatten(feats):
    if len(feats.size()) == 2:
        return feats
    feats = feats.view(feats.size(0), feats.size(1), -1).transpose(2, 1)\
            .contiguous().view(-1, feats.size(1))
    return feats 
def get_init_centroids(in_dim, seed, K, featlist, index):
    clus = faiss.Clustering(in_dim, K)
    clus.seed  = np.random.randint(seed)
    clus.niter = 100000 #fix
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


    num_init_batches = 32
    in_dim = 32
    K_train = 20


    kmeans_loss  = AverageMeter()
    faiss_module = get_faiss_module(in_dim=in_dim)
    data_count   = np.zeros(K_train)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            img = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
            sal = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
            indices = batch['key']['meta']['index']
            indices = indices.long().cpu().numpy()
            
            feats, _ = model.model_k(img) 

            bsz = feats.shape[0]
            dim = feats.shape[1]
            feats = nn.functional.normalize(feats, dim=1)
            feats = feats.reshape(bsz, dim, -1) # B x dim x H.W

            sal = sal.reshape(bsz, -1, 1).type(feats.dtype)


            #feats = F.normalize(feats, dim=1, p=2)
            #  feats = feature_flatten(feats).detach().cpu()


            feats = torch.bmm(feats, sal).squeeze() # B x dim
            feats = nn.functional.normalize(feats, dim=1)  # Bx dim
            
            
    
            
            if i_batch == 0:
                print('Batch input size : {}'.format(list(img.shape)))
                print('Batch feature : {}'.format(list(feats.shape)))
            
            feats = feats.detach().cpu()

            if num_batches < num_init_batches:
                featslist.append(feats)
                num_batches += 1
                
                if num_batches == num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
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
                    num_batches = num_init_batches - num_batches

            if (i_batch % 100) == 0:
                print('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))
    centroids = torch.tensor(centroids, requires_grad=False).cuda()

    return centroids, kmeans_loss.avg

def get_metric_as_conv(centroids, device):
    N, C = centroids.size()

    centroids_weight = centroids.unsqueeze(-1).unsqueeze(-1)
    metric_function  = nn.Conv2d(C, N, 1, padding=0, stride=1, bias=False)
    metric_function.weight.data = centroids_weight
    metric_function = metric_function.to(device)
    
    return metric_function

def compute_negative_euclidean(featmap, centroids, metric_function):
    centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    return - (1 - 2*metric_function(featmap)\
                + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 

def postprocess_label(p, K, idx, idx_img, scores, n_dual):
    out = scores[idx].topk(1, dim=0)[1].flatten().detach().cpu().numpy()

    # Save labels. 
    # if not os.path.exists(os.path.join(save_model_path, 'label_' + str(n_dual))):
        # os.makedirs(os.path.join(args.save_model_path, 'label_' + str(n_dual)))
    # torch.save(out, os.path.join(args.save_model_path, 'label_' + str(n_dual), '{}.pkl'.format(idx_img)))
    
    # Count for re-weighting. 
    counts = torch.tensor(np.bincount(out, minlength=K)).float()

    return counts

def compute_labels(p, logger, dataloader, model, centroids, device):
    """
    Label all images for each view with the obtained cluster centroids. 
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    K = centroids.size(0)

    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids, device)

    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
            sal = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
            indices = batch['query']['meta']['index']
            
            feats, _ = model.model_q(img) 

            B, C, H, W = feats.shape
            if i == 0:
                print('Centroid size      : {}'.format(list(centroids.shape)))
                print('Batch input size   : {}'.format(list(img.shape)))
                print('Batch feature size : {}\n'.format(list(feats.shape)))
 

            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(feats, centroids, metric_function) 

            # Save labels and count. 
            for idx, idx_img in enumerate(indices):
                counts += postprocess_label(K, idx, idx_img, scores, n_dual=view)

            if (i % 200) == 0:
                print('[Assigning labels] {} / {}'.format(i, len(dataloader)))
    weight = counts / counts.sum()
        
    return weight
            

def do_kmeans(p, dataloader, model):
    centroids, kmloss = run_mini_batch_kmeans(p, dataloader, model)
    

def train(p, N, train_loader, model, optimizer, epoch, amp):
    losses = AverageMeter('Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive', ':.4e')
    saliency_losses = AverageMeter('CE', ':.4e')
    superpixel_losses = AverageMeter('Superpixel', ':.4e')


    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), 
                        [losses, contrastive_losses, superpixel_losses, saliency_losses, top1, top5],
                        prefix="Epoch: [{}]".format(epoch))
    model.train()

    if p['freeze_layers']:
        model = freeze_layers(model)
    
    #
    do_kmeans(p, train_loader, model)

    # All feat
    all_features, all_indices = get_all_features(p, train_loader, model, N) # N x dim 
    # Cluster all 
    all_assignments = cluster_all_features(all_features, n_clusters=20)
    
    # Anchor feat 
    anchor_features = get_anchor_features(all_features, all_assignments) # C x dim
    anchor_features = torch.from_numpy(anchor_features).float().cuda(p['gpu'], non_blocking=True)
    pseudo_dataset = PseudoDataset(all_indices, all_assignments)

    for i, batch in enumerate(train_loader):
        # Forward pass
        im_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
        sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
        im_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
        sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
        indices = batch['query']['meta']['index']
        
        pseudo_labels = pseudo_dataset[indices]
        pseudo_labels = torch.from_numpy(pseudo_labels).long()

        superpixel_logits, superpixel_labels, logits, labels, saliency_loss = model(im_q=im_q, sal_q=sal_q, im_k=im_k, sal_k=sal_k, anchor=anchor_features, pseudo_labels=pseudo_labels)


        # print(logits.shape)
        # print(labels.shape)
        # print(torch.unique(labels))

        # Use E-Net weighting for calculating the pixel-wise loss.
        # uniq, freq = torch.unique(labels, return_counts=True)
        # p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
        # p_class_non_zero_classes = freq.float() / labels.numel()
        # p_class[uniq] = p_class_non_zero_classes
        # w_class = 1 / torch.log(1.02 + p_class)
        # contrastive_loss = cross_entropy(logits, labels, weight=w_class,
        #                                     reduction='mean')
        contrastive_loss = cross_entropy(logits, labels,
                                            reduction='mean')

        superpixel_loss = cross(superpixel_logits, superpixel_labels, reduction='mean')

    #     # Calculate total loss and update meters
        loss = contrastive_loss + saliency_loss + superpixel_loss
        
        contrastive_losses.update(contrastive_loss.item())
        saliency_losses.update(saliency_loss.item())
        superpixel_losses.update(superpixel_loss.item())

        losses.update(loss.item())
        
        


        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        top1.update(acc1[0], im_q.size(0))
        top5.update(acc5[0], im_q.size(0))
        

        # Update model
        optimizer.zero_grad()
        if amp is not None: # Mixed precision
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()            
        else:
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
    writer.add_scalar('superpixel loss', superpixel_losses.avg, epoch)
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
