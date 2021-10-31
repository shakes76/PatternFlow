import os

import torch

from model import ModelSize, YOLOModel


##########################################################
#   Constants
##########################################################

# where the cache saves
model_data_path = os.path.join("snapshot", "yolox.pth")

# for use we only have 1 class
classes = ["lesion"]
num_classes = len(classes)

# Must be multiple of 32
input_shape = (512, 512)

# Default Network Size
model_size: ModelSize = ModelSize.S

# Suppress candidate boxes below this confidence
threshold = 0.5

# For NMS, the higher it goes, the less boxes it detect
iou = 0.8

# Turn on GPU or not
device = torch.device("cuda:0")

# How many CPU threads required
# tune up if CPU is the hurdle
# tune down if no enough ram
num_of_worker = 8


class YOLOX:
    def __init__(self) -> None:
        self.net = YOLOModel(num_classes, model_size)
        self.net.load_state_dict(torch.load(model_data_path, map_location=device))
        self.net = self.net.eval()
