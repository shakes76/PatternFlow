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
   3. The coordinate format of our created bounding box is (x_min, y_min, x_max, y_max) to be in line with most used dataset COCO
      - I planned to pretrain our model on COCO, but the benefit of pretraining YOLOX is quite low([YOLOX][yolox2021]: Strong data augmentation), so I skip it

## Data augmentation

The original YOLOX use Mosaic + Mix Up to boost generalization performance, but in our task, there is only one category to classify, and having both augmentation may cause negative impact on performance ([YOLOX: Table 5][yolox2021])
So we use Scale Jit + Mosaic only

## Model - YOLOX

YOLOX is one of latest work in YOLO Big Family, it utilize an anchor-free approach with a decoupled head. To put it simple, it's a YOLO v3 + Anchor Free approach.

The original YOLOX model repo is published on [GitHub](https://github.com/Megvii-BaseDetection/YOLOX).
But this model is writen in pyTorch, while I am using Tensorflow(for better visualization), I also customized Image dimension

## Network - CSPNet

The CSPNet is firstly introduced by the controversial YOLO v5
### Loss function

Why choose AP over AOC? Because in YOLO, we will have many negative cases, in ROC curve true positive is equivalently important as true negative. So a module missed the only bounding box may still get a great AOC score, that's definitely not what we want.
mAP is 

### Tensorflow Keras

The reason to chose tensorflow over pyTorch.
1.  TensorBoard. I used the TensorboardX + pyTorch in Lab 2, but it's not very satisfactory, so I want to try tensorflow this time. 
2.  Tensorflow recently added Functional API that shows similar flexibility of pyTorch functional API.
3.  Tensorflow have a mobile library called Tensorflow Lite, I want to deploy this API in one of the other project.

## Results


## Reference

- [Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian, YOLOX: Exceeding YOLO Series in 2021. arXiv preprint arXiv:2107.08430][yolox2021]

[yolox2021]: https://arxiv.org/pdf/2107.08430.pdf "Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian, YOLOX: Exceeding YOLO Series in 2021. arXiv preprint arXiv:2107.08430"

## Appendix: Challenges faced

My task never run in Goliath servers.
The dataset is relatively large. So it takes time to upload to any paid service

