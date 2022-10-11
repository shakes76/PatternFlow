from torch import embedding, nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = ['resnet18', 'resnet34', 'resnet50']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, num_features=128, num_classes=2):
        super(ResNet, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = num_features
        
        # Construct base (pretrained) resnet

        resnet = ResNet.__factory[depth](pretrained=True)

        self.out_planes = resnet.fc.in_features

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool)

        self.fc1 = nn.Linear(self.out_planes, self.num_features)
        self.fc1_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.fc1.weight, mode='fan_out')
        init.constant_(self.fc1.bias, 0)
        self.fc2 = nn.Linear(self.num_features, self.num_classes)
        init.kaiming_normal(self.fc2.weight, mode='fan_out')
        init.constant_(self.fc2.bias, 0)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.base(x)
        emb = x.view(x.size(0), -1)
        prob = self.fc2(self.relu(self.fc1_bn(self.fc1(emb))))
        return emb, prob


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)
