# A MASK R-CNN network for lesion detection within the ISIC Dataset.
This is a tensorflow 2.2 implimentation of MASK R-CNN model to solve report promblem 3.

## Usage
 - python==3.7.11
 - numpy==1.20.3
 - scipy==1.4.1
 - Pillow==8.4.0
 - cython==0.29.24
 - matplotlib
 - scikit-image==0.16.2
 - tensorflow==2.2.0
 - keras==2.3.1
 - opencv-python==4.5.4.60
 - h5py==2.10.0
 - imgaug==0.4.0
 - IPython[all]

*All the libraries required is in the requirement.txt*

*Note that it won't run on any Apple Silicon devices due to the old library required*


## MASK R-CNN 

### What is MASK R-CNN?
MASK R-CNN is an object detection model developed by a AI research group in Facebook in 2017 developed on Fast R-CNN, which is based on deep convolutional neural netwworks (CNN). The model can return both the bounding box and a mask for each detected object in images or videos.

### How does it work?
To understand Mask R-CNN, let's first discus architecture of Faster R-CNN that works in two stages:

 - Stage 1: The first stage consists of two networks, backbone (ResNet, VGG, Inception, etc..) and region proposal network. These networks run once per image to give a set of region proposals. Region proposals are regions in the feature map which contain the object.
 - Stage 2: In the second stage, the network predicts bounding boxes and object class for each of the proposed region obtained in stage1. Each proposed region can be of different size whereas fully connected layers in the networks always require fixed size vector to make predictions. Size of these proposed regions is fixed by using either RoI pool (which is very similar to MaxPooling) or RoIAlign method.

 ![how mask r-cnn works](/documentation/how_maskrcnn_works.png)

Faster R-CNN predicts object class and bounding boxes. Mask R-CNN is an extension of Faster R-CNN with additional branch for predicting segmentation masks on each Region of Interest (RoI).


## The Dataset
The dataset used in the model is based on the ISIC 2017 dataset, containing:
 - 200 annotated images for training and validation, with coco-like labels
 - 14 images for testing

Annotation is done by hand using the tool provided in [Makesense.ai](https://www.makesense.ai/).

*(Doing the annotation is really a pain in the ass tbh)*

Here's a picture showing the pre-annotated image:

![pre-annotated image](/documentation/Figure_1.png)


## Files
 - 'dataset.py' containing the data loader for loading and preprocessing
 - 'modules.py' containing the source code of the components of the model
 - 'train.py' containing the source code for training, validating, testing and saving the model. Saved model can be found in the 'logs' folder in '.h5' format after you run the training.
 - 'predict.py' showing test results of the trained model.


## Model Training
In my training, the model was trained for 10 epochs, each containg 100 steps.
Each epoch takes eoughly 5min on my 2070max-q.

Pre-trained coco weights are used, and you can download it on this [link](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5).
And here's the [link](https://github.com/matterport/Mask_RCNN) to the original repo from matterport that works on TF1.x.

### My Training Results
```
100/100 [==============================] - 347s 3s/step - loss: 1.2868 - rpn_class_loss: 0.0121 - rpn_bbox_loss: 0.4415 - mrcnn_class_loss: 0.0457 - mrcnn_bbox_loss: 0.4698 - mrcnn_mask_loss: 0.3177 - val_loss: 0.9349 - val_rpn_class_loss: 0.0074 - val_rpn_bbox_loss: 0.3950 - val_mrcnn_class_loss: 0.0236 - val_mrcnn_bbox_loss: 0.2386 - val_mrcnn_mask_loss: 0.2702
Epoch 2/10
100/100 [==============================] - 290s 3s/step - loss: 0.7364 - rpn_class_loss: 0.0087 - rpn_bbox_loss: 0.2950 - mrcnn_class_loss: 0.0285 - mrcnn_bbox_loss: 0.1982 - mrcnn_mask_loss: 0.2060 - val_loss: 0.7340 - val_rpn_class_loss: 0.0055 - val_rpn_bbox_loss: 0.2969 - val_mrcnn_class_loss: 0.0256 - val_mrcnn_bbox_loss: 0.1864 - val_mrcnn_mask_loss: 0.2196
Epoch 3/10
100/100 [==============================] - 294s 3s/step - loss: 0.6204 - rpn_class_loss: 0.0061 - rpn_bbox_loss: 0.2387 - mrcnn_class_loss: 0.0281 - mrcnn_bbox_loss: 0.1659 - mrcnn_mask_loss: 0.1816 - val_loss: 0.6144 - val_rpn_class_loss: 0.0063 - val_rpn_bbox_loss: 0.2614 - val_mrcnn_class_loss: 0.0248 - val_mrcnn_bbox_loss: 0.1497 - val_mrcnn_mask_loss: 0.1722
Epoch 4/10
100/100 [==============================] - 300s 3s/step - loss: 0.5416 - rpn_class_loss: 0.0057 - rpn_bbox_loss: 0.2234 - mrcnn_class_loss: 0.0288 - mrcnn_bbox_loss: 0.1195 - mrcnn_mask_loss: 0.1641 - val_loss: 0.5549 - val_rpn_class_loss: 0.0040 - val_rpn_bbox_loss: 0.2706 - val_mrcnn_class_loss: 0.0199 - val_mrcnn_bbox_loss: 0.1118 - val_mrcnn_mask_loss: 0.1486
Epoch 5/10
100/100 [==============================] - 296s 3s/step - loss: 0.4889 - rpn_class_loss: 0.0050 - rpn_bbox_loss: 0.2038 - mrcnn_class_loss: 0.0274 - mrcnn_bbox_loss: 0.1017 - mrcnn_mask_loss: 0.1510 - val_loss: 0.7551 - val_rpn_class_loss: 0.0073 - val_rpn_bbox_loss: 0.4620 - val_mrcnn_class_loss: 0.0383 - val_mrcnn_bbox_loss: 0.1058 - val_mrcnn_mask_loss: 0.1418
Epoch 6/10
100/100 [==============================] - 297s 3s/step - loss: 0.4702 - rpn_class_loss: 0.0048 - rpn_bbox_loss: 0.2122 - mrcnn_class_loss: 0.0280 - mrcnn_bbox_loss: 0.0890 - mrcnn_mask_loss: 0.1363 - val_loss: 0.4313 - val_rpn_class_loss: 0.0055 - val_rpn_bbox_loss: 0.1611 - val_mrcnn_class_loss: 0.0309 - val_mrcnn_bbox_loss: 0.0792 - val_mrcnn_mask_loss: 0.1547
Epoch 7/10
100/100 [==============================] - 306s 3s/step - loss: 0.4153 - rpn_class_loss: 0.0049 - rpn_bbox_loss: 0.1667 - mrcnn_class_loss: 0.0269 - mrcnn_bbox_loss: 0.0849 - mrcnn_mask_loss: 0.1319 - val_loss: 0.4132 - val_rpn_class_loss: 0.0056 - val_rpn_bbox_loss: 0.1755 - val_mrcnn_class_loss: 0.0252 - val_mrcnn_bbox_loss: 0.0753 - val_mrcnn_mask_loss: 0.1316
Epoch 8/10
100/100 [==============================] - 303s 3s/step - loss: 0.4021 - rpn_class_loss: 0.0050 - rpn_bbox_loss: 0.1625 - mrcnn_class_loss: 0.0265 - mrcnn_bbox_loss: 0.0741 - mrcnn_mask_loss: 0.1340 - val_loss: 0.3603 - val_rpn_class_loss: 0.0039 - val_rpn_bbox_loss: 0.1396 - val_mrcnn_class_loss: 0.0221 - val_mrcnn_bbox_loss: 0.0702 - val_mrcnn_mask_loss: 0.1245
Epoch 9/10
100/100 [==============================] - 319s 3s/step - loss: 0.3687 - rpn_class_loss: 0.0044 - rpn_bbox_loss: 0.1420 - mrcnn_class_loss: 0.0253 - mrcnn_bbox_loss: 0.0716 - mrcnn_mask_loss: 0.1254 - val_loss: 0.3182 - val_rpn_class_loss: 0.0031 - val_rpn_bbox_loss: 0.1254 - val_mrcnn_class_loss: 0.0193 - val_mrcnn_bbox_loss: 0.0565 - val_mrcnn_mask_loss: 0.1139
Epoch 10/10
100/100 [==============================] - 307s 3s/step - loss: 0.3364 - rpn_class_loss: 0.0042 - rpn_bbox_loss: 0.1335 - mrcnn_class_loss: 0.0212 - mrcnn_bbox_loss: 0.0587 - mrcnn_mask_loss: 0.1188 - val_loss: 0.3134 - val_rpn_class_loss: 0.0046 - val_rpn_bbox_loss: 0.1114 - val_mrcnn_class_loss: 0.0183 - val_mrcnn_bbox_loss: 0.0557 - val_mrcnn_mask_loss: 0.1233
```

And the test result looks like this:

![test image](/documentation/output1.png)

![test result](/documentation/output.png)

You can see from the image that the lesion is successfully picked out by the model.