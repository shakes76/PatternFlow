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

def parse_data(inputs):
    imgs, labels, indexes = inputs
    return imgs.cuda(), labels.cuda(), indexes.cuda()
