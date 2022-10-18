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
from modules import resnet18, resnet34, resnet50