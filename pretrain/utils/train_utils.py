#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
from torch.nn.functional import cross_entropy
from torchvision import transforms
from utils.utils import AverageMeter, ProgressMeter, freeze_layers


def train(p, train_loader, model, optimizer, epoch, amp):
    losses = AverageMeter('Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive', ':.4e')
    consistency_losses = AverageMeter('Consistency', ':.4e')
    #local_losses = AverageMeter('Local', ':.4e')
    #iic_losses = AverageMeter('ICC', ':.4e')
    #cluster_losses = AverageMeter('Cluster', ':.4e')
    #entropy_losses = AverageMeter('Entropy', ':.4e')
    saliency_losses = AverageMeter('CE', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), 
                        [losses, contrastive_losses, consistency_losses, saliency_losses, top1, top5],
                        prefix="Epoch: [{}]".format(epoch))
    model.train()

    if p['freeze_layers']:
        model = freeze_layers(model)

    for i, batch in enumerate(train_loader):
        # Forward pass
        im_q = batch['query']['image'].cuda(p['gpu'], non_blocking=True)
        im_k = batch['key']['image'].cuda(p['gpu'], non_blocking=True)
        sal_q = batch['query']['sal'].cuda(p['gpu'], non_blocking=True)
        sal_k = batch['key']['sal'].cuda(p['gpu'], non_blocking=True)
        state_dict =  batch['T'].cuda(p['gpu'], non_blocking=True)

        logits, labels, saliency_loss, consistency_loss = model(im_q=im_q, im_k=im_k, sal_q=sal_q, sal_k=sal_k, state_dict=state_dict)
      
        # Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = cross_entropy(logits, labels, weight=w_class,
                                            reduction='mean')
      
        ## Calculate local contrastive loss
        # uniq_local, freq_local = torch.unique(l_labels, return_counts=True)
        # p_class_local = torch.zeros(l_logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
        # p_class_non_zero_classes_local = freq_local.float() / l_labels.numel()
        # p_class_local[uniq] = p_class_non_zero_classes_local
        # w_class_local = 1 / torch.log(1.02 + p_class_local)
        # local_loss = cross_entropy(l_logits, l_labels, weight= w_class_local,reduction='mean')



        # Calculate total loss and update meters
        #loss = contrastive_loss + saliency_loss + 0.1*(cluster_loss - 5*entropy + 0.01 * upper_clamp + 0.01 * lower_clamp )
        loss = contrastive_loss + saliency_loss + consistency_loss 
        contrastive_losses.update(contrastive_loss.item())
        #cluster_losses.update(cluster_loss.item())
        # local_losses.update(local_loss.item())
        #iic_losses.update(iic_loss.item())   
        saliency_losses.update(saliency_loss.item())
        consistency_losses.update(consistency_losses.item())
        #entropy_losses.update(entropy.item())

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
            save_plot_curve(contrastive_losses=contrastive_losses, 
                            saliency_losses=saliency_losses,
                            # cluster_losses=cluster_losses,
                            # entropy_losses= entropy_losses,
                            #iic_losses = iic_losses,
                            consistency_losses=consistency_losses,
                            losses=losses
                            )
       

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
    contrastive_losses, saliency_losses, consistency_losses, losses,
    path = '/content/drive/MyDrive/UCS_local/pretrained_result/VOCSegmentation_supervised_saliency_model/'):

    with open(path+'cl.txt', 'a') as f:
        f.write(str(contrastive_losses.avg))
        f.write("\n")
    with open(path+'consistency.txt', 'a') as f:
        f.write(str(consistency_losses.avg))
        f.write("\n")
    # with open(path+'localcl.txt', 'a') as f:  
    #     f.write(str(local_losses.avg))
    #     f.write("\n")
    # with open(path + 'iic.txt', 'a') as f:
    #     f.write(str(iic_losses.avg))
    #     f.write("\n")
    # with open(path + 'cluster.txt', 'a') as f:
    #     f.write(str(cluster_losses.avg))
    #     f.write("\n")
    with open(path+'saliency.txt', 'a') as f:
        f.write(str(saliency_losses.avg))
        f.write("\n")
    # with open(path+'entropy.txt', 'a') as f:
    #     f.write(str(entropy_losses.avg))
    #     f.write("\n")
    with open(path+'total.txt', 'a') as f:
        f.write(str(losses.avg))
        f.write("\n")
    