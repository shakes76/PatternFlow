import random
import torch
import numpy as np
from torch.backends import cudnn
from torch import nn
import torchvision
import torch
from torch import nn
import torch.nn.functional as F



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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



def pdist(v):
    dist = torch.norm(v[:, None] - v, dim=2, p=2)
    return dist


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # pairwise distances
        dist = pdist(inputs)

        # find the hardest positive and negative
        mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_neg = ~mask_pos
        mask_pos[torch.eye(n).byte().cuda()] = 0

        # weighted sample pos and negative to avoid outliers causing collapse
        posw = (dist + 1e-12) * mask_pos.float()
        posi = torch.multinomial(posw, 1)
        dist_p = dist.gather(0, posi.view(1, -1))
        # There is likely a much better way of sampling negatives in proportion their difficulty, based on distance
        # this was a quick hack that ended up working better for some datasets than hard negative
        negw = (1 / (dist + 1e-12)) * mask_neg.float()
        negi = torch.multinomial(negw, 1)
        dist_n = dist.gather(0, negi.view(1, -1))

        # calculate loss
        diff = dist_p - dist_n
        if isinstance(self.margin, str) and self.margin == 'soft':
            diff = F.softplus(diff)
        else:
            diff = torch.clamp(diff + self.margin, min=0.)
        loss = diff.mean()

        return loss



def parse_data(inputs):
    imgs, labels, indexes = inputs
    return imgs.cuda(), labels.cuda(), indexes.cuda()