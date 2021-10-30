# Pattern Recognition Report - Task 3 with YOLOX

## Identity

> Student ID: 45742785

## Abstract

what we did for this report is training a [YOLOX][yolox2021] model on ISIC 2018 Task 1-2 Dataset, to clearly recognize the lesion area.
Result in mAP with minimum IoU

## Data preprocessing

1. Training Image: 
   1. None. Because we are going to do advanced augmentation on the image, we don't have to do too much here
2. Annotation:
   1. Firstly we need to apply the same transformation to the annotation.
   2. Draw the bounding box according to the shape of silhouette(first and last non-zero row/column)
   3. The coordinate format of our created bounding box is (xmin, ymin, xmax, ymax) to be in line with most used dataset COCO
      - I planned to pretrain our model on COCO, but the benefit of pretraining YOLOX is quite low([YOLOX][yolox2021]: Strong data augmentation), so I skip it

## Data augmentation

The original YOLOX use Mosaic + Mix Up to boost generaliziton performance, but in our task, there is only one category to classify, and having both augmentation may cause negative impact on performance ([YOLOX: Table 5][yolox2021])
So we use Scale Jit + Mosaic only

## Model - YOLOX

YOLOX is one of latest work in YOLO Big Family, it utilize an anchor-free approach with a decoupled head. To put it simple, it's a YOLO v3 + Anchor Free approach.

The original YOLOX model repo is published on [GitHub](https://github.com/Megvii-BaseDetection/YOLOX).
But this model is writen in pyTorch, while I am using Tensorflow(for better visualization), I also customized Image dimention

## Network - CSPNet

The CSPNet is firstly introduced by the controversial YOLO v5

## Driver - Tensorflow Keras

## Results

## Reference

- [Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian, YOLOX: Exceeding YOLO Series in 2021. arXiv preprint arXiv:2107.08430][yolox2021]

[yolox2021]: https://arxiv.org/pdf/2107.08430.pdf "Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian, YOLOX: Exceeding YOLO Series in 2021. arXiv preprint arXiv:2107.08430"

## Apendix: Challenges faced

My task never run in Goliath servers.
The dataset is relatively large. So it takes time to upload to any paid service

