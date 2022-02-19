#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from audioop import cross
import torch
from torch.nn.functional import cross_entropy
from utils.utils import AverageMeter, ProgressMeter, freeze_layers
from torch.utils.tensorboard import SummaryWriter, writer 
import os
import torch.nn as nn 

def get_all_features(p, train_loader, model, N):
   
    for i, batch in enumerate(train_loader):
        # Forward pass
        img = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
        sal = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)

        bsz = img.shape[0]

        feat, _ = model.model_q(img) 
        feat = nn.functional.normalize(feat, dim=1)
        feat = feat.reshape(feat, feat.shape[1], -1) # B x dim x H.W
        feat = feat.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        feat = torch.reshape(feat, [-1, feat.shape[-1]]) # queries: pixels x dim
        sal = sal.reshape(bsz, -1, 1).type(feat.dtype)

        feat_mean = torch.bmm(feat, sal).squeeze() # B x dim
        feat_mean = nn.functional.normalize(feat_mean, dim=1)  # Bx dim
        
        if i == 0:
            all_feat = torch.zeros((N, feat_mean.shape[-1]))
        
        if i < len(train_loader) - 1:
            all_feat[i * bsz: (i + 1) * bsz] = feat_mean
        else:
            all_feat[i * bsz:] = feat_mean
    
    return all_feat

def train(p, train_loader, model, optimizer, epoch, amp):
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
    
    # All feat
    all_featues = get_all_features(p, train_loader, model)

    # for i, batch in enumerate(train_loader):
    #     # Forward pass
    #     im_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
    #     im_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
    #     sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
    #     sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)

    #     logits, labels, obj_logits, obj_labels, saliency_loss = model(im_q=im_q, im_k=im_k, sal_q=sal_q, sal_k=sal_k)

    #     # Use E-Net weighting for calculating the pixel-wise loss.
    #     uniq, freq = torch.unique(labels, return_counts=True)
    #     p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
    #     p_class_non_zero_classes = freq.float() / labels.numel()
    #     p_class[uniq] = p_class_non_zero_classes
    #     w_class = 1 / torch.log(1.02 + p_class)
    #     contrastive_loss = cross_entropy(logits, labels, weight=w_class,
    #                                         reduction='mean')



    #     superpixel_loss = cross_entropy(obj_logits, obj_labels, reduction='mean')



    #     # Calculate total loss and update meters
    #     loss = contrastive_loss + saliency_loss + superpixel_loss
        
    #     contrastive_losses.update(contrastive_loss.item())
    #     saliency_losses.update(saliency_loss.item())
    #     superpixel_losses.update(superpixel_loss.item())

    #     losses.update(loss.item())
        
        


    #     acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
    #     top1.update(acc1[0], im_q.size(0))
    #     top5.update(acc5[0], im_q.size(0))
        

    #     # Update model
    #     optimizer.zero_grad()
    #     if amp is not None: # Mixed precision
    #         with amp.scale_loss(loss, optimizer) as scaled_loss:
    #             scaled_loss.backward()            
    #     else:
    #         loss.backward()
    #     optimizer.step()

    #     # Display progress
    #     if i % 25 == 0:
    #         progress.display(i)
    
    # writer_path = os.path.join(p['output_dir'], "runs")
    # writer = SummaryWriter(log_dir=writer_path)
    # writer.add_scalar('total loss', losses.avg, epoch)
    # writer.add_scalar('contrastive loss', contrastive_losses.avg, epoch)
    # writer.add_scalar('saliency loss', saliency_losses.avg, epoch)
    # writer.add_scalar('superpixel loss', superpixel_losses.avg, epoch)
    # writer.close()      


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
