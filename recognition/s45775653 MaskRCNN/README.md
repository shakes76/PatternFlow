# ISICs 2018 Leison Segmentation using Mask R-CNN

This project uses [Mask R-CNN](https://arxiv.org/abs/1703.06870) to predict lesion segmentation boundaries within dermoscopic images [ISIC 2018 Challenge - Task 1: Lesion Boundary Segmentation](https://challenge.isic-archive.com/landing/2018/45/).Mask-RCNN is an extension of [Faster R-CNN](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) where it also uses a region proposal network (RPN) that proposes candidate bounding boxes at the first stage, and modifies the second stage where the same RoIPool is used to extract features then adds a parallel generates an output for each RoI [1]. Mask R-CNN optimizes the three (3) loss functions for each ROI namely, classification loss (_Lcls_), bounding-box class loss (_Lbox_), and mask loss (_Lmask_). The total loss (_L_) is defined as _L = Lcls + Lbox + Lmask_. The goal of this project is to acheive an Intersection over Union (IoU) score of 0.8 for the images.

## ISICs 2018 Challenge: Lesion Boundary Segmentation
The ISICs Lesion Boundary segmentation pre-processed dataset contains a total of 2,594 Lesion images (in .jpg format) with accompanying mask segmentations (in .png format). For the mask segmentations the values 0 and <255 represent the background, and 255 represents the target object. The figure below shows the Lesion image with its accompanying mask.

![Lesion Image + Mask](https://github.com/christianburbon/lettuce_annotation/blob/master/other_images/visualize_image_mask.png)

## Data Pre-processing
