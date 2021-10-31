# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:45:43 2021

@author: christian burbon
"""
#%%

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import os
import glob
import cv2
import skimage.io as skio
import sys
import time
from PIL import Image, ImageDraw
import gc
import imgaug.augmenters as iaa
import imgaug.parameters as iap
#%%

# Load Mask RCNN Library
ROOT_DIR = 'D:/Python/mask_rcnn_aktwelve/Mask_RCNN/'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
import data_splitter as ds
#%%
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


#%%
# Create Config

class ISICConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'isic_train'

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 1 (leison mask)

    # All of our training images are resized to 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Set to 1500 since images are 1400+
    STEPS_PER_EPOCH = 1500

    # This is how often validation is run.
    VALIDATION_STEPS = 50
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50' #'resnet101'
    IMAGE_RESIZE_MODE = "square"
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56,56)
    
config = ISICConfig()
config.display()


#%%

# Define the dataset loader class

class ISICDataset(utils.Dataset):
    '''
    Genenerates keras based dataset for the images and masks
    '''
    
    def load_data(self,masks_dir,images_dir):
        '''
        Generates a dictionary of the images for easy calling using the keras based dataset
        parameters:
            masks_dir: the exact filepath to where the ground truth mask images
            images_dir: the exact filepath to where the images are stored
        '''
        self.masks_dir = masks_dir
        self.images_dir = images_dir
        # Define the filetypes to search for
        img_spec = '*.jpg' # train images
        
        # populate image dictionary
        img_names = []
        for imgfile in os.listdir(images_dir):
            if imgfile.endswith('.png') or imgfile.endswith('.jpg') or imgfile.endswith('.jpeg'):
                img_names.append(imgfile[:-4])
                
                image_id = imgfile[:-4]
                image_file_name = imgfile
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                
                self.add_image(source="isic",
                               image_id=image_id,
                               path=image_path)
                self.add_class("isic",1,"leison")
   
    
    def load_mask(self, image_id):
        '''
        parameters:
            image_id: takes the image file name from load_data
        
        returns:
            mask: ground truth mask image formatted for keras based datasets
            class_ids: returns the mask class_id
        '''
        seg_spec = '*.png' # gt images
        seg_names = []
        instance_masks = []
        class_ids = []
        
        image_info = self.image_info[image_id]

        filename = glob.glob(os.path.join(self.masks_dir,image_info['id']+"*"))

        mask_seg = cv2.imread(filename[0],0)

        bool_mask = mask_seg>0
        instance_masks.append(bool_mask)
        class_ids.append(1)

        mask = np.dstack(instance_masks)
        # mask = instance_masks
        class_ids = np.array(class_ids,dtype=np.int32)
        
        return mask, class_ids
#%%

# Split the dataset and create augmentations

# Define train image source. This is the original dataset
train_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'

# Define segmentation mask source. This is the original dataset
seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'

# Define train, val, test, and test full (full resolution) image, and mask folder save locations
# These are where the processed images will be saved
train_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/train_img'
train_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/train_seg'
val_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/val_img'
val_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/val_seg'
test_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_img'
test_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_seg'
test_full_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_img_full'
test_full_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_seg_full'


# Generate the augmentations, and save to respective destination folders. Takes time due to augmentations.
# You can adjust the max_res, and resize factor depending on your GPU
ds.datasplitter(train_path,seg_path,
             train_img_path,train_seg_path,
             val_img_path,val_seg_path,
             test_img_path,test_seg_path, 
             test_full_img_path,test_full_seg_path, 
             ,max_res=700*700,resize=0.25)

print('Done')
#%%

# Prepare the Train, Validation, Test datasets. Test set uses full resolution images, others use augmented images.

dataset_train = ISICDataset()
dataset_train.load_data(train_seg_path,train_img_path)
dataset_train.prepare()

dataset_val = ISICDataset()
dataset_val.load_data(val_seg_path,val_img_path)
dataset_val.prepare()

dataset_test = ISICDataset()
dataset_test.load_data(test_full_seg_path,test_full_img_path)
dataset_test.prepare()
#%%

# Visualize the dataset and ensure that images, and ground truth masks are loaded properly

dataset = dataset_test
image_ids = np.random.choice(dataset.image_ids,4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

#%%

gc.collect()
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
print('model generated')
#%%

# Load Pre-trained weights

def load_weights(init_with="coco",model_path=None):
    '''
    choose pre-trained weights to start using init_with
    parameters:
        init_with: a string value in ["coco","last","specifi"] to start the weights with. Default = "coco"
            coco: Load weights trained on MS COCO, but skip layers that are different due to the different number of classes
            last: Load the last model you trained and continue training
            specific: load a specific model passing the path of the model.h5 file within logs folder.
                sample path = 'isic_train20211029T1127/mask_rcnn_isic_train_0030.h5'
    '''
# Which weights to start with?
    if init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
        print('coco weights loaded')
        
    elif init_with == "last":
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)
        print(model_path)
    
    elif init_with == "specific":
        modelh5_path = model_path
        model.load_weights(os.path.join(MODEL_DIR,modelh5_path))
        print('model loaded from',modelh5_path)
        
load_weights()
#%%

# Create Augmentation Strategy - Use only non-geometric
augment_strat = iaa.SomeOf((0,None),[
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
    iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
    iaa.Sharpen(alpha=0.5)
    ])

gc.collect()
#%%
def scheduler_(epoch, lr = config.LEARNING_RATE):
    '''
    Define the learning rate (lr) scheduler using the lr specified in config
    '''
    if epoch < 15:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
custom_callback = keras.callbacks.LearningRateScheduler(scheduler_)  
#%%
# Training                        
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
start_train = time.time()

#LAYERS
              # heads: The RPN, classifier and mask heads of the network
              # all: All the layers
              # 3+: Train Resnet stage 3 and up
              # 4+: Train Resnet stage 4 and up
              # 5+: Train Resnet stage 5 and up

# Train model using heads layer first to learn some features
model.train(dataset_train
            ,dataset_val
            ,learning_rate=config.LEARNING_RATE
            ,epochs=5
            ,layers='heads'
            ,custom_callbacks = [custom_callback]
            ,augmentation = augment_strat
            )

# Train model with deeper layers. Maximum that my device can handle is 3+
model.train(dataset_train
            ,dataset_val
            ,learning_rate=config.LEARNING_RATE
            ,epochs=30
            ,layers='3+'
            ,custom_callbacks = [custom_callback]
            ,augmentation = augment_strat
            )

end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')

#%%

# Set the config for Inferencing mode

gc.collect()

class InferenceConfig(ISICConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56,56)
    
inference_config = InferenceConfig()
#%%
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
print('Inference Mode')
#%%

def inference_weight(model_path=None):
    '''
    parameters:
        model_path = the file path to load as specific model from.
        sample model path when != None: logs/isic_train20211029T1127/mask_rcnn_isic_train_0025.h5
    '''
    if model_path != None:
        model_path = os.path.join(MODEL_DIR,model_path)
    else: # get the latest one from logs
        model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
inference_weight(model_path=None)

#%%
import skimage

dataset = dataset_test
image_ids = np.random.choice(dataset.image_ids,5)
for image_id in image_ids:
    img = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    
    # run model detection
    results = model.detect([img], verbose=1)
    r = results[0]
    
    # display the predicted mask
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_train.class_names, r['scores'], figsize=(10,15))
    
    # display the original image & the original mask
    plt.figure(figsize=(10,15))
    plt.subplot(1,2,1);plt.imshow(img)
    plt.subplot(1,2,2);plt.imshow(mask,cmap='gray')

#%%

# mean AP
dataset = dataset_test
iou_threshold=0.80
# Compute VOC-style Average Precision
def compute_batch_ap(image_ids):
    '''
    Parameters:
        image_ids: the image_id generated from the dataset
    Returns:
        APs: List of the mean Average Precision (mAP) of the images.
        precisions_all: List of the average precisions per image at different class score thresholds.
        recall_all: List of the average recall values per image at different class score thresholds.
        iou_all: List of individual object's max IoU score across every image.
    
    see mrcnn utils.py "compute_ap" for more details
    '''
    
    APs = []
    precisions_all = []
    recall_all = []
    iou_all = []

    for image_id in image_ids:

        print(image_id)
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config, image_id)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        res_mask = utils.minimize_mask(r['rois'],r['masks'],config.MINI_MASK_SHAPE)
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], res_mask,iou_threshold=iou_threshold)
        # visualize.plot_precision_recall(AP, precisions, recalls)
        APs.append(AP)
        precisions_all.append(np.mean(precisions))
        recall_all.append(np.mean(recalls))
        if np.size(overlaps) == 0:
            iou_all.append(0)
        else:
            iou_all.append(np.max(overlaps))
        
        
    return APs, precisions_all, recall_all, iou_all

image_ids = dataset.image_ids
APs, precisions, recalls, ious = compute_batch_ap(image_ids)
ious_ = np.array(ious)
print("mAP @IoU="+str(iou_threshold)+": ", np.round(np.mean(APs),4))
print("model IoU:", np.round(np.mean(ious),4))
print("images >= 80% IoU:",len(ious_[ious_>=0.8])," images < 80% IoU:",len(ious_[ious_<0.8]))
plt.boxplot(ious);plt.title("Image IoU Scores");plt.tick_params(labelbottom=False)