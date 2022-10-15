
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model

import numpy as np
'''
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import argparse
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import os
import numpy as np
import matplotlib.pyplot as plt
import random

print("TF Version:", tf.__version__)