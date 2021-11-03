#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
from kornia.geometry import transform
from numpy import matrix
import torch
from torch.nn.functional import cross_entropy
from torchvision import transforms
from utils.utils import AverageMeter, ProgressMeter, freeze_layers


def train(p, train_loader, model, optimizer, epoch, amp):
    losses = AverageMeter('Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive', ':.4e')
    inveqv_losses = AverageMeter('Inveqv', ':.4e')
    saliency_losses = AverageMeter('CE', ':.4e')
    mean_losses = AverageMeter('Mean-Contrast', ':.4e')
    spatial_losses = AverageMeter('Spatial', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), 
                        [losses, contrastive_losses, inveqv_losses, saliency_losses, mean_losses, spatial_losses, top1, top5],
                        prefix="Epoch: [{}]".format(epoch))
    model.train()
    

    if p['freeze_layers']:
        model = freeze_layers(model)

    for i,  batch in enumerate(train_loader):
        # Forward pass
        
        im_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
        im_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
        sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
        sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
        
        if p['loss_coeff']['inveqv'] > 0:
            im_ie = batch['inveqv']['image'].cuda(p['gpu'], non_blocking=True)
            sal_ie = batch['inveqv']['sal'].cuda(p['gpu'], non_blocking=True)
        else:
            im_ie = batch['inveqv']['image']
            sal_ie = batch['inveqv']['sal']
        
        matrix_eqv = batch['matrix']
        size_eqv = batch['size']

            
        
        logits, labels, saliency_loss, inveqv_loss, m_logits, m_labels, spatial_loss = model(im_q=im_q, im_k=im_k, sal_q=sal_q, sal_k=sal_k, im_ie=im_ie, sal_ie=sal_ie, matrix_eqv=matrix_eqv, size_eqv=size_eqv, dataloader=train_loader)
        

        
        # Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = cross_entropy(logits, labels, weight=w_class,
                                            reduction='mean')
        

        

        if p['loss_coeff']['mean'] > 0:
            uniq_mean, freq_mean = torch.unique(m_labels, return_counts=True)
            p_class_mean = torch.zeros(m_logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
            p_class_non_zero_classes_mean = freq_mean.float() / m_labels.numel()
            p_class_mean[uniq_mean] = p_class_non_zero_classes_mean
            w_class_mean = 1 / torch.log(1.02 + p_class_mean)
            mean_loss = cross_entropy(m_logits, m_labels, weight= w_class_mean, reduction='mean')
        else:
            mean_loss = 0.
        
        # Calculate total loss and update meters
        loss = p['loss_coeff']['contrastive'] * contrastive_loss +\
                p['loss_coeff']['saliency'] * saliency_loss + \
                p['loss_coeff']['inveqv']* inveqv_loss +\
                p['loss_coeff']['mean'] * mean_loss +\
                p['loss_coeff']['spatial'] * spatial_loss
                 
        

        contrastive_losses.update(contrastive_loss.item())

        if p['loss_coeff']['mean'] > 0:
            mean_losses.update(mean_loss.item())
        else:
            mean_losses.update(mean_loss)
         
        if p['loss_coeff']['inveqv'] > 0:
            inveqv_losses.update(inveqv_loss.item())
        else:
            inveqv_losses.update(inveqv_loss)

        if p['loss_coeff']['saliency'] > 0:
            saliency_losses.update(saliency_loss.item())
        else:
            saliency_losses.update(saliency_loss)
        
        if p['loss_coeff']['spatial'] > 0:
            spatial_losses.update(spatial_loss.item())
        else:
            spatial_losses.update(spatial_loss)

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
            
    save_plot_curve(
        p=p,
        contrastive_losses=contrastive_losses,
        saliency_losses=saliency_losses,
        inveqv_losses=inveqv_losses,
        mean_losses=mean_losses,
        losses=losses,
        spatial_losses=spatial_losses
    )
    return losses.avg

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
    

def save_plot_curve(
    p,
    contrastive_losses, 
    saliency_losses,
    inveqv_losses,
    mean_losses,
    losses,
    spatial_losses
    ):

    with open(os.path.join(p['output_dir'], 'cl.txt'), 'a') as f:
        f.write(str(contrastive_losses.avg))
        f.write("\n")
    with open(os.path.join(p['output_dir'], 'inveqv.txt'), 'a') as f:
        f.write(str(inveqv_losses.avg))
        f.write("\n")
    with open(os.path.join(p['output_dir'], 'saliency.txt'), 'a') as f:
        f.write(str(saliency_losses.avg))
        f.write("\n")
    with open(os.path.join(p['output_dir'], 'total.txt'), 'a') as f:
        f.write(str(losses.avg))
        f.write("\n")
    with open(os.path.join(p['output_dir'], 'mean-contrast.txt'), 'a') as f:
        f.write(str(mean_losses.avg))
        f.write("\n")
    with open(os.path.join(p['output_dir'], 'spatial.txt'), 'a') as f:
        f.write(str(spatial_losses.avg))
        f.write("\n")