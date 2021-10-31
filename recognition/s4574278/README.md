# Pattern Recognition Report - Task 3 with YOLOX

## Identity

> Student ID: 45742785

## Abstract

what we did for this report is training a simplified [YOLOX][yolox2021] model on ISIC 2018 Task 1-2 Dataset, to clearly recognize the lesion area.

## Introduction
<!-- TODO: put Result in mAP with minimum IoU -->

## Data preprocessing

1. Training Image: 
   1. None. Because we are going to do advanced augmentation on the image, we don't have to do too much here
2. Annotation:
   1. Firstly we need to apply the same transformation to the annotation.
   2. Draw the bounding box according to the shape of silhouette(first and last non-zero row/column)
   3. The coordinate format of our created bounding box is (x_min, y_min, x_max, y_max) to be in line with most used dataset COCO
      - I planned to pretrain our model on COCO, but the benefit of pretraining YOLOX is quite low([YOLOX][yolox2021]: Strong data augmentation), so I skip it

### Data augmentation

The original YOLOX use Mosaic + Mix Up to boost generalization performance, but in our task, there is only one category to classify, and having too much augmentation may cause negative impact on performance\[[1][yolox2021]\]
So we plan to use only the basic Scale Jit. Mosaic is too troublesome for me to do, besides, we can use simplier model when the input is also simple. 

## Model - YOLOX

YOLOX is one of the latest work in YOLO Family, it is built on top of YOLO v3, utilize an anchor-free approach and combined with recent research progress on Deep Learning, like: decoupled head, SimOTA, Mosaic Data Augmentation, etc. Comparing to YOLO v5, it might be slower in some cases, but the AP is largely improved.

The original YOLOX model repo is published on [GitHub](https://github.com/Megvii-BaseDetection/YOLOX).
But this model is written in pyTorch, while I am using Tensorflow(reasons are stated later), I also customized a bit to fit our use case.

Some commenter say YOLOX, due to its Anchor free nature, it more similar to "[FCOS](tian2019fcos)".

### Backbone - CSPNet\[[3][wang2019cspnet]\]

The CSPNet is firstly introduced into YOLO family by YOLO v4. To me, it looks like a modified ResNet. From the dense prediction of YOLO v1 to modified CSPNet in YOLO v5 and YOLOX, the capacity of feature extraction is drastically improved along the way.

### Activation

According to the paper[\[1\]][yolox2021], We used Sigmoid Linear Units, or SiLUs, it looks like a ReLU but smooth.
![https://paperswithcode.com/method/silu](./images/SiLU.png)
Image from [Sigmoid Linear Unit | paperswithcode.com](https://paperswithcode.com/method/silu)

### Loss function

CIoU


Why choose AP over AOC(Area under ROC-curve)? Because in YOLO, we will generate many anchor boxes(anchor points in YOLOX) most of them are negative cases and should be cancelled anyway. In ROC curve true positive is equivalently important as true negative. So in this unbalanced scenario, a model which missed the only bounding box may still get a pretty decent AOC score, that's definitely not what we want.
mAP on the other hand, emphasize on the positive case.

## Decision Justification
### Tensorflow Keras

The reason to chose tensorflow over pyTorch.
1.  TensorBoard. I used the TensorboardX + pyTorch in Lab 2, but it's not very satisfactory, so I want to try tensorflow this time. 
2.  Tensorflow recently added Functional API that shows similar flexibility of pyTorch functional API.
3.  Tensorflow have a mobile library called Tensorflow Lite, I want to deploy this API in one of the other project.

## Results


## Reference

<!-- https://www.bibtex.com/c/bibtex-to-ieee-converter/ -->
1. [Z. Ge, S. Liu, F. Wang, Z. Li, en J. Sun, “YOLOX: Exceeding YOLO Series in 2021”, arXiv [cs.CV]. 2021.][yolox2021]
2. [A. Bochkovskiy, C.-Y. Wang, en H.-Y. M. Liao, “YOLOv4: Optimal Speed and Accuracy of Object Detection”, arXiv [cs.CV]. 2020][bochkovskiy2020yolov4]
3. [C.-Y. Wang, H.-Y. M. Liao, I.-H. Yeh, Y.-H. Wu, P.-Y. Chen, en J.-W. Hsieh, “CSPNet: A New Backbone that can Enhance Learning Capability of CNN”, arXiv [cs.CV]. 2019.][wang2019cspnet]
4. [Z. Tian, C. Shen, H. Chen, en T. He, “FCOS: Fully Convolutional One-Stage Object Detection”, arXiv [cs.CV]. 2019.][tian2019fcos]
5. [S. Elfwing, E. Uchibe, en K. Doya, “Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning”, arXiv [cs.LG]. 2017.][elfwing2017sigmoidweighted]
6. [Dan Hendrycks and Kevin Gimpel, Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:][hendrycks2020gaussian]

[yolox2021]: https://arxiv.org/abs/2107.08430 "YOLOX: Exceeding YOLO Series in 2021"
[bochkovskiy2020yolov4]: https://arxiv.org/abs/2004.10934 "YOLOv4: Optimal Speed and Accuracy of Object Detection"
[wang2019cspnet]: https://arxiv.org/abs/1911.11929 "CSPNet: A New Backbone that can Enhance Learning Capability of CNN"
[tian2019fcos]: https://arxiv.org/abs/1904.01355 "FCOS: Fully Convolutional One-Stage Object Detection"
[elfwing2017sigmoidweighted]: https://arxiv.org/abs/1702.03118v3 "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning"
[hendrycks2020gaussian]: https://arxiv.org/abs/1606.08415 "Gaussian Error Linear Units (GELUs)"

## Appendix: Challenges faced

- My queued task never run in Goliath servers. So I booked a paid GPU service.
- The dataset is relatively large. So it takes time to upload it to the server. Especially if you need to transfer the archive to another country like United States, may takes days to complete. *AVOID Paperspace*

