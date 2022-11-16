from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw

def detect_image(model_conf_path,img_size,weights_path,img ,conf_thres,nms_thres ):
    Tensor = torch.cuda.FloatTensor     
    model = Darknet(model_conf_path, img_size=img_size)
    model.load_weights(weights_path)
    model.cuda()
    model.eval()
    # create scales and padding data to convert the input image to a square shape 416*416
    width, height = img.size
    max_v = max(height, width)
    ratio = 416 / max_v
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio) 
    #Tensor = torch.cuda.FloatTensor    
    if imh>imw: 
        dim_diff =imh - imw
        pad_left, pad_right = dim_diff // 2, dim_diff - dim_diff // 2
        pad=(pad_left,0,pad_right,0)
    elif imw>imh:
        dim_diff =imw - imh
        pad_top, pad_bottom = dim_diff // 2, dim_diff - dim_diff // 2
        pad=(0,pad_top,0,pad_bottom)
    
 #this is the padding for the left, top, right and bottom borders respectively.
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad(pad,0),
         transforms.ToTensor(),
         ])
    # resize and pad image and convert it to tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
   
    # run inference on the darmnet model and use OASIS trained weight and get detections
    with torch.no_grad():
        detections = model(input_img)
    # call this function to filter best bounding box out of multiple detected bounding boxes        
        detections = utils.non_max_suppression(detections, 1, conf_thres, nms_thres)
    return detections[0]