import random
import torch
import numpy as np
from torch.backends import cudnn
from torch import nn
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from dataset import ADNI
from modules import resnet34
from utils import AverageMeter, pdist, parse_data


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
        diff = F.softplus(diff)
        # if isinstance(self.margin, str) and self.margin == 'soft':
        #     diff = F.softplus(diff)
        # else:
        #     diff = torch.clamp(diff + self.margin, min=0.)
        loss = diff.mean()

        return loss

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = True

    height = 224
    width = 224
    num_features = 128
    num_classes = 2
    num_instances = 256
    batch_size = num_instances * num_classes
    learning_rate = 1e-4
    weight_decay = 1e-4
    margin = 0.3
    lamda = 1.0
    num_epochs = 500

    pretraining=True

    dataset = ADNI("./AD_NC")
    train_loader = dataset.get_train_loader(height, width, batch_size)
    test_loader = dataset.get_test_loader(height, width, 2048)

    model = resnet34(num_features=num_features, num_classes=num_classes, pretraining=pretraining)
    model.cuda()

    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    criterion_triplet = TripletLoss(margin=margin)

    train_loss = []
    test_acc = []

    best_acc = -1

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        losses = AverageMeter()
        model.train()

        for step, inputs in enumerate(train_loader):
            imgs, labels, indexes = parse_data(inputs)

            emds, probs = model(imgs)

            loss_triplet = criterion_triplet(emds, labels)
            loss_cla = F.cross_entropy(probs, labels)
            loss = loss_cla + loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

        train_loss.append(losses.avg)
        print("Loss {:.3f}".format(losses.avg))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for step, inputs in enumerate(test_loader):
                imgs, labels, indexes = parse_data(inputs)
                emds, probs = model(imgs)
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct / total
            test_acc.append(acc)
            if best_acc <= acc:
                best_acc = acc
                torch.save(model.state_dict(), "best_model.pkl")

        print("test acc {:.3f}, best acc {:.3f}".format(acc, best_acc))




