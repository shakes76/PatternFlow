#import required dependencies and libraries

from __future__ import division
from models import *
from train import *
from utils.utils import *
from utils.preprocess import *
from utils.parse_config import *
from models import *
from detection import *

import os, sys, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import torch.optim as optim


#set the configuration parameters for re-training Darknet pretrained model on OASIS dataset

model_conf_path="config/yolov3.cfg" # " the path of the config file of the darknet pretrained model"
data_conf_path="config/OASIS.data" # "  data config file path"
classnames_path="config/OASIS.names" # " class names file path"
weights_path="config/yolov3.weights" # " this is the path to the weights of pre-trained darknet model which is downloaded from web"
checkpoint_interval=1# "interval between saving model weights"
checkpoint_dir="checkpoints"# "directory where model checkpoints are saved, after the model trained for the specified number of epochs, the saved weights from this directory is used for model inference"
use_cuda=True# "whether to run the model on gpu or not"
n_cpu=0 # "number of cpu threads to use during batch generation"


#initialize the model parameters 
epochs=20 # "number of epochs"
img_data="data/ISIS/images" #"path to images dataset"
batch_size=16 #help="size of image batch"
img_size = 416 # we need to resize the image data to square with size 416*416


train_model(epochs,img_data,batch_size,model_conf_path,data_conf_path,weights_path,classnames_path,n_cpu,img_size,checkpoint_interval,checkpoint_dir,use_cuda)

#the output of this function call is the weights which are generated and saved in checkpoint folder.


#in order to run the model in reference mode and run object detection, we need to chnage the weight config of the model. we need to use the weights which are generate during model training
weights_path='checkpoints/19.weights'
class_path='config/OASIS.names'
#this is the path of the weight file generated from last epoch of model training


#initialize the Non Max Suppression parameter to filter the detected bounding boxes
conf_thres=0.9 # confidence threshold. Any box that has a confidence below this threshold will be removed.
nms_thres=0.3  #threshold for IOU . This threshold is used to remove boxes that have a high overlap
img_path = "data/ISIS/images/ISIC_0014805_segmentation.png" # the path of the test image for object detection

img = Image.open(img_path)
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
# Load model and weights
# Load model and weights
# model = Darknet(model_conf_path, img_size=img_size)
# model.load_weights(weights_path)
# model.cuda()
# model.eval()
detections = detect_image(model_conf_path,img_size,weights_path,img,conf_thres,nms_thres)
width, height = img.size
max_v = max(height, width)
ratio = img_size / max_v
imw = round(img.size[0] * ratio)
imh = round(img.size[1] * ratio)     
if imh>imw: 
  dim_diff =imh - imw
  pad_left, pad_right = dim_diff // 2, dim_diff - dim_diff // 2
  pad=(pad_left,0,pad_right,0)
elif imw>imh:
  dim_diff =imw - imh
  pad_top, pad_bottom = dim_diff // 2, dim_diff - dim_diff // 2
  pad=(0,pad_top,0,pad_bottom)

img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad(pad,0),
         transforms.ToTensor(),
         transforms.ToPILImage()
         ])
image_pil = img_transforms(img)
if detections is not None:
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # browse detections and draw bounding boxes
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        draw = ImageDraw.Draw(image_pil)
        draw.rectangle(((x1,y1), (x2,y2)),outline="white", width=3)
image_pil


#filter the bounding boxes further to select the boxes with maximum class probabolities
#filter detected bounding box with maximum class score
def bbox_max_class_trs(detection):
    value, index =torch.max(detections,  0)
    return detection[index[4],0:4]

#show the image with the selected bounding box
image_target = img_transforms(img)
x1, y1, x2, y2 = bbox_max_class_trs(detections)
draw = ImageDraw.Draw(image_target)
draw.rectangle(((x1,y1), (x2,y2)),outline="white", width=3)
image_target
