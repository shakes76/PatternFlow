# ISICs 2018 Leison Segmentation using Mask R-CNN

This project uses [Mask R-CNN](https://arxiv.org/abs/1703.06870) [1] to predict lesion segmentation boundaries within dermoscopic images [ISIC 2018 Challenge - Task 1: Lesion Boundary Segmentation](https://challenge.isic-archive.com/landing/2018/45/) [2] . The Mask R-CNN model will be trained with the goal of acheiving an Intersection over Union (IoU) score of 0.8 for the predicted image segmentation.

## ISICs 2018 Challenge: Lesion Boundary Segmentation
The ISICs Lesion Boundary segmentation pre-processed dataset contains a total of 2,594 Lesion images (in .jpg format) with accompanying mask segmentations (in .png format). For the mask segmentations the pixel values 0 upto less than 255 represent the background, and 255 represents the target object. The Figure 1 shows the Lesion image with its accompanying mask.

![dataset1](https://github.com/christianburbon/isic_maskrcnn_copy/blob/master/visualize_dataset/imgmask_1.png)

Figure 1: Lesion Images + Accompanying Masks

## Data Pre-processing
Before any pre-processing of the data, to ensure that there is no data leakage among the augmentations of the original image and the original image itself, data is split into Training, Validation, and Test at the beginning. First, data is split into Train (75%), and Test (25%), and second the Train set is further split into Train (75%), and Validation (25%).
One of the biggest challenges in CNNs is the amount of computation required to process large resolution images. In order to overcome this challenge, the images were resized to 25% of original when it exceed 700*700 pixels, otherwise, resolution is retained. Additionally, to account for varying positions of the Lesion in images, Images are flipped vertical, and horizontal separately. Images are normalized by dividing them by 255. Lastly, the target object is then labelled as "leison". The pre-processing sequence is shown in Figure 2, and the flipped images, and masks are shown in Figure 3. After augmentation, the total images per dataset are as follows: Train Set (4,374), Validation Set (1,461), Test Set (649).

![Image Pre-processing](https://github.com/christianburbon/lettuce_annotation/blob/master/other_images/pre-processing.jpg)

Figure 2: Image Pre-processing Steps


![pre-processed images](https://github.com/christianburbon/isic_maskrcnn_copy/blob/master/pre_processing.png)

Figure 3: Image and Mask After Pre-processing


## Mask R-CNN Architecture

Mask-RCNN is an extension of [Faster R-CNN](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) [3] where it also uses a region proposal network (RPN) that proposes candidate bounding boxes during the first stage, and then modifies the second stage where the same RoIPool is used to extract features then adds a parallel generates an output for each RoI. Mask R-CNN optimizes the three (3) loss functions for each ROI namely, classification loss (_Lcls_), bounding-box class loss (_Lbox_), and mask loss (_Lmask_). The total loss (_L_) is defined as _L = Lcls + Lbox + Lmask_ [1]. The Mask RCNN architecture is show in Figure 4.

![mask rcnn](https://github.com/christianburbon/isic_maskrcnn_copy/blob/master/other_images/mask_rcnn%20architecture.png)

Figure 4: Mask R-CNN Architecture [1]


## Training Configuration and Results
### Configuration
The model was trained using pre-trained model weights using [COCO](https://cocodataset.org) dataset. The training follows a two-step procedure that takes advantage of selecting the depth of the model to decrease/increase amount of feature learning. Two layer configurations were used, namely, the _"heads"_ (The RPN, classifier and mask heads of the network) and _"3+"_ (_heads_ + Train Resnet upto stage 3) layers. The "heads" layers on the first 5 epochs, and then "3+" layers upto the 30th epoch. The dataset is shuffled every epoch, and 1500 steps were taken for each. Non-geometric augmentations are also applied during training where none upto all of the augmentations are used randomly (Figure 5).

![Train Augmentations](https://github.com/christianburbon/isic_maskrcnn_copy/blob/master/other_images/training_augmentations.png)

Figure 5: Train Augmentations


### Training Loss
Figure 6 shows the overall training loss after each epoch, other loss metrics are shown in the model_loss folder. Note that step 0 is the 1st epoch, and step 29 is the 30th.

![Epoch Loss](https://github.com/christianburbon/isic_maskrcnn_copy/blob/master/model_loss/epoch_loss.png)

Figure 6: Overall Training Loss


### Prediction Results
The goal of the project is to reach an IoU score of >= 0.80, and this project has reached 0.85 on the test set where Figure 7 shows the IoU score distribution per image. It is known that there is only 1 object per image, and to account for the possibility of having detected multiple instances, the maximum IoU score per image was taken.



## References
[1] He, Kaiming, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. (Jan 2018). ‘Mask R-CNN’. Facebook AI Research (FAIR). [Online] Available:http://arxiv.org/abs/1703.06870.
[2] 
[3] 
