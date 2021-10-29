# ISICs 2018 Leison Segmentation using Mask R-CNN

This project uses [Mask R-CNN](https://arxiv.org/abs/1703.06870) to predict lesion segmentation boundaries within dermoscopic images [ISIC 2018 Challenge - Task 1: Lesion Boundary Segmentation](https://challenge.isic-archive.com/landing/2018/45/). The Mask R-CNN model will be trained with the goal of acheiving an Intersection over Union (IoU) score of 0.8 for the predicted image segmentation.

## ISICs 2018 Challenge: Lesion Boundary Segmentation
The ISICs Lesion Boundary segmentation pre-processed dataset contains a total of 2,594 Lesion images (in .jpg format) with accompanying mask segmentations (in .png format). For the mask segmentations the pixel values 0 upto less than 255 represent the background, and 255 represents the target object. The Figure 1 shows the Lesion image with its accompanying mask.

![Lesion Image + Mask](https://github.com/christianburbon/lettuce_annotation/blob/master/other_images/visualize_image_mask.png)

Figure 1: Lesion Images + Accompanying Masks

## Data Pre-processing
One of the biggest challenges in CNNs is the amount of computation required to process large resolution images. In order to overcome this challenge, the images were resized to 25% of original when it exceed 700*700 pixels, otherwise, resolution is retained. Additionally, to account for varying positions of the Lesion in images, Images are flipped vertical, and horizontal separately. Images are normalized by dividing them by 255. Lastly, the target object is then labelled as "leison". The pre-processing sequence is shown in Figure 2, and the flipped images, and masks are shown in Figure 3.

![Image Pre-processing](https://github.com/christianburbon/lettuce_annotation/blob/master/other_images/pre-processing.jpg)

Figure 2: Image Pre-processing Steps

Insert Fig3 here

Figure 3: Images After Pre-processing


## Mask R-CNN Architecture

Mask-RCNN is an extension of [Faster R-CNN](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) where it also uses a region proposal network (RPN) that proposes candidate bounding boxes during the first stage, and then modifies the second stage where the same RoIPool is used to extract features then adds a parallel generates an output for each RoI. Mask R-CNN optimizes the three (3) loss functions for each ROI namely, classification loss (_Lcls_), bounding-box class loss (_Lbox_), and mask loss (_Lmask_). The total loss (_L_) is defined as _L = Lcls + Lbox + Lmask_ [1].
