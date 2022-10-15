# TODO:
- problems section
  - what are the problems
  - show results, plots, images, etc
  - what was done to try solve this - include results, plots, images, etc
- reproducability of results? - batch size on different machine
- proper formatting?
- albumentations - ?
- which branch to pull request - topic-recognition
# Object Detection with YOLO in the ISIC Dataset
Please note that module.py is empty because the model itself is inside of the yolov5_LC git submodule
## Dependencies:
### YOLOv5 Dependencies
This YOLOv5 implementation contains quite a few dependencies, however, once the YOLOv5 submodule is added, the installation process is quite simple. Firstly, the submodule must be added by executing the following command in the s4532810-YOLO-xspinella directory.

```linux
git submodule update --init
```
After this, all the dependencies required by YOLOv5 can be installed with the following commands:

```linux
cd yolov5_LC
pip install -r requirements.txt
```

### PatternFlow-xspinella Dependencies
For data downloading/preprocessing/arranging, the following additional dependencies are required:

```linux
pip install gdown zipfile36 
```

## Usage
### Setup
Once the dependencies are installed, execute the following command in the s4532810-YOLO-xspinella directory:

```linux
python3 train.py
```
and select mode 0 from the options to download/arrange/preprocess the data completely. 

### Training and Evaluation
The model can be trained by executing 
```linux
python3 train.py
```
again, and selecting mode 2, or mode 1 can be selected to train and test the model in one step. However, it is reccomended to train (mode 2), then test (mode 3) separately, so that you can see the training results. 
If mode 2 or 1 is selected, the program will ask for the yolo model size, and requires the user to enter in one of the following characters:
n - YOLOv5n -> 'nano'
s - YOLOv5s -> 'small'
m - YOLOv5m -> 'medium'
l - YOLOv5l -> 'large'
x - YOLOv5x -> 'extra large'
The yolov5 github specifies the quantitative differences between the models, but basically, as the model gets larger, it will become slower, but perform better.
See this link for specifications: https://github.com/ultralytics/yolov5#pretrained-checkpoints
It is reccomended to use the medium size model, as this is the model investigated in this report.

### Inference
In order to deploy the trained model, predict.py is available. predict.py contains the Predictor class, and a Predictor_Example_Use() function, which shows the user how to load the model, and produce and visualise object detection/classification on a single image. It also shows the user how to visualise comparisons between predicted and labelled object detection/classification and how to use utils_lib to compute IOU and find if the classification is correct (if the user wishes to run comparisons on labelled sets).

## Problem definition:
### A Word on the ISIC dataset.
ISIC is an organisation that aims to "Support efforts to reduce melanoma-related deaths and unnecessary biopsies by improving the accuracy and efficiency of melanoma early detection." The organisation aims to do this by engaging computer science communities to improve diagnostic accuracy, with the aid of AI techniques. ISIC 2017 will be used in this project - it contains 2000 training images, 150 validation images, and 600 test images. Each of these images contains a lesion which is classified as either: [melanoma, seborrheic_keratosis], [!melanoma, seborrheic_keratosis], [melanoma, !seborrheic_keratosis], or [!melanoma, !seborrheic_keratosis]  (https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/aboutIsicGoals).

### Project Scope
This implementation defines the object detection problem as the detection of skin lesions, and classification of melanomas, i.e. the classification of seborrheic keratosis is out of scope.

## YOLO:
This section will detail What the algorthm is, what problem it solves, and how it works.
### Description of YOLO
The YOLO model family consists of multiple different versions of an object detection model. Object detection models aim to search for object classes within given images. Once found, these objects are indicated by a bounding box, and classified with a label. YOLO stands for You Only Look Once, because each frame (or image) is only passed through the model once. The result of this is that YOLO models are smalller and faster  than other object detection implementations, which traditionally passed the image through once for defining bounding boxes, and then again to classify the box classes. (https://blog.roboflow.com/a-thorough-breakdown-of-yolov4/)

### YOLO Applications
YOLO models excel in realtime object detection, thanks to their but fast performance. These models are also very lightweight, which not only means that they can be implemented on video feeds at relatively high frame rate, they can also be deployed on native hardware easier than other models, because they do not require as much computing power. This project aims to use a YOLO model to detect and classify lesions within the ISIC dataset. (https://blog.roboflow.com/a-thorough-breakdown-of-yolov4/)

### How YOLO Works
The YOLO model is so successful because it frames object detection as a single regression problem; mapping image pixels to bounding box coordinates and class probabilities. The system works by splitting the input into an S x S grid (S is a hyperparameter). If the centre of an object falls inside box 2 x 2, then this box is responsible for detecting said object. When implemented, each grid cell will predict B bounding boxes, where B is also a hyperparameter. Each computed bounding box is comprised of 5 values:

![image](https://user-images.githubusercontent.com/32262943/193556475-503dec60-c9d1-4135-a001-7a910bee09ea.png)

x, y: The x/y-coordinate of the bounding box centre, relative to the bounds of the grid cell.

w, h: The width and height of the bounding box, relative to the whole image.

C: The confidence score, which is both a measure of how confident the model is that this bounding box contains an object, and how accurate it thinks the predicted box is. If trained perfectly, this value should be equal to the Intersection Over Union (IOU) between the predicted box, and the ground truth, if there is one in the grid cell of interest. If there is no ground truth in the grid cell, this value should be 0. When implemented, the contained object confidence value is multipled by the IOU box accuracy value:  
![image](https://user-images.githubusercontent.com/32262943/193551577-41be3605-3038-4d9a-999f-c1fe5cabb0bb.png)

The loss function evaluated during training is as follows:

![image](https://user-images.githubusercontent.com/32262943/193552858-933318ae-473a-4766-8f8c-243e365df288.png)

This is a multi-part loss function which specifies terms for defining the loss of 

- Bounding box location and size (the first two terms)
- Confidence predictions for boxes that contain objects, and boxes that dont contain objects (third and fourth terms, respectively)
- Classification predictions for the objects inside the boxes (the last term).

These terms are balanced by parameters \lambda_coord and \lambda_noobj to ensure that confidence scores from grid cells without objects don't overpower the gradient from cells which do contain objects. 

A high-level overview of the YOLO model architechture is shown below (https://arxiv.org/pdf/2004.10934.pdf).
![image](https://miro.medium.com/max/720/1*e17LeKXUsSxdNTlSm_Cz8w.png)

The backbone is the pretrained component, while the head is trained to predict the actual bonding boxes and classes on the dataset of interest. As shown above, the head can be single (just dense prediction) or double stage (dense and sparse prediction). The neck, situated between backbone and head, is used to collect feature maps for the dataset (https://medium.com/analytics-vidhya/object-detection-algorithm-yolo-v5-architecture-89e0a35472ef).

### YOLOv5 network model
For this implementation, I have chosen to use YOLOv5 because it is written in Pytorch ultralytics, while previous versions were written in the C Darknet library. YOLOv5 is also faster than previous versions, and while there have been newer YOLO releases since v5, YOLOv5 currently has the most support. 

## Metrics of Interest
In the discussion of results, there are a few metrics which will be referred to. The following sections will explain their meanings.

### Intersection Over Union (IOU)
This metric is used to measure how closely the bounding box prediction matches the bounding box ground truth. It is defined as:

![image](https://user-images.githubusercontent.com/32262943/194858311-10f53308-65e9-49ce-a47c-228579b3f27e.png)

Which could also be interpretted as:

IOU = overlap area/[(pred area + gnd truth area) - overlap area]

(https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

In object detection, an IOU threshold is used to define whether a bounding box prediction is either a False Positive (FP), or True Positive (TP) i.e. if the IOU for a box predicition is above the threshold, the box prediction is considered a TP, and vice versa. A False Negative (FN) is when no box is predicted, but it should have been, and a True Negative (TN) is when no box is predicted, and the ground truth also specifies no box (i.e. no object to detect). (https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

### Precision
Precision is a measure of how accurate box predictions are, and uses the IOU threshold to specifiy TPs, TNs, FPs, FNs. It is defined as:

![image](https://user-images.githubusercontent.com/32262943/194860335-b9d6bb78-213d-4ad8-86c6-df8f293540dd.png)

From this formula, we can see that Precision could also be defined as "the percentage of box predictions which were correct." One should note that this metric doesn't account for FNs. (https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

### Recall
Recall Also uses the IOU threshold, and is a measure of how many objects were detected successfully. It can be defined as below:

![image](https://user-images.githubusercontent.com/32262943/194861794-10382fab-5037-409f-87a6-68f9f55bd07c.png)

This formula shows that while recall measures the percentage of objects which were detected successfully, it does not account for FPs. (https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

### Precision-Recall relationship
The relationship between precision and recall is often a trade-off, and the favourable metric often depends on the use case. In the case of melanoma detection, we would rather detect too many objects (FPs) than not detect enough objects (FNs), because a surplus of FNs could result in an increase of missed melanoma diagnoses. Thus, for this project application, we would favour recall over precision, and would accept the trade-off of increased FPs.  

### mAP
AP or Average Precision is the area under the Precision-Recall curve, and is used across most object detection models as a measure of how well the model draws boundary boxes. In the result plots, both mAP0.5 and mAP0.5:0.95 are reported. The numbers simply refer to IOU threshold -> mAP0.5 is just the AP at IOU-T=0.5, while mAP0.5:0.95 is the average of all the AP values for IOU-T values between 0.5 and 0.95, with a step size of 0.05 (https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

### Box Loss
A regression loss bosed on the calculatd IOU -> the bounding box term of the loss function. Smaller value implies better boxes.

### Object Loss
A loss based on the confidence score. Smaller value means more accurate confidence scores, which implies that the model is better at predicting when there is an object in the image, and where it is. A smaller value also indicates that, during deployment, users of the model can more accurately judge whether they should trust the model prediction. 

### Class Loss
This loss is a measure of how well the model classifies objects. However, it should be noted that this loss metric is only penalises classification
error if an object is present in the current grid cell of interest, i.e. if the model predicts the class of a box, but the centre does not fall in the same grid cell as the ground truth box, it is not penalised/rewarded for this prediction.

## Dataloader Functionality
This library is designed such that any user can pull it, add the dependencies (shown in above section), then simply execute the following:

```linux
python3 train.py
```
and select mode 0 from the options to download/arrange/preprocess the data completely - that is, after running the dataloader (mode 0), the ISIC dataset is completely ready for training.

### File arrangement
The Create_File_Structure method in Dataloader checks which directories don't exist, and creates them all. This includes directories inside of yolov5_LC which weren't pushed from my machine due to git ignore. After image preprocessing, Copy_Images and Copy_Configs are used to copy the yolo-compatible dataset, labels, and .yaml training/testing config files to the correct directories within the yolov5_LC directory.

### Downloads
The ISIC dataset, and all its gnround truth and classification files are downloaded and placed in the correct directories with Download_Zips and Extract_Zips. Unwanted superpixel images are removed by Delete_Unwanted_Files.

### Preprocessing
The dataset is preprocessed in 4 different ways:
- Resize to 640x640 with torchvision.transforms.resize (implemented in the Resize_Images Dataloader member)
- Ground truth segmentations are converted to box specification (Cx, Cy, w, h) with the utils_lib function Mask_To_Box
- Classification csvs are converted into class labels (0 for not melanoma, 1 for melanoma) with the Find_Class_From_CSv utils_lib function
- YOLOv5-format txt file labels are created for each image in the dataset with Dataloader member Create_YOLO_Labels
- The Training/Validation/Test split is as per the ISIC 2017 dataset.

## Model Optimisation
### Problems Faced
Once all the data was preprocessed and setup as per YOLOv5 training specifications, the model was trained on YOLOv5m for 450 epochs, but stopped early at ~250 epochs, as it had stopped improving. The training data can be seen below:

Figure 1:

![image](https://user-images.githubusercontent.com/32262943/195742748-9b17c73f-b8ff-4faa-8fd5-644e411be1f5.png)

Straight away, this raises a few issues:
- While the validation box loss decreases, the validation objectness loss begins to increase at the end, and 
- The validation classification loss seems to be very noisey, but has an increasing trend for seemingly the entire training session.

After conducting some research into the matter, it appears that this is generally due to a lack of data - documentation mentions that the dataset should have at least 10000 labelled objects per class (https://docs.ultralytics.com/tutorials/training-tips-best-results/), however, as shown by the below graph, the ISIC dataset has considerably less than this:

Figure 2:

![image](https://user-images.githubusercontent.com/32262943/195752043-48c03638-40e1-49af-8c71-6a682955353f.png)

After running the model on train.py's test mode (3), the following results were produced:

Figure 3:

![image](https://user-images.githubusercontent.com/32262943/195755190-5a6ba9bc-a8b8-40b1-b3ce-6e011a89e1e4.png)

Figure 4:

![image](https://user-images.githubusercontent.com/32262943/195755278-2115333b-0154-44f7-a6a2-c74af10f7ab2.png)

This reveals the following problems:
- mAP0.5:0.95 not as high as it should be for yolov5m -> should be 45.2, but is 38.9
- The average IOU is not at the satisfactory level of 0.8

An interesting observation here is that the classification accuracy is above the acceptable value of 0.8, even though the classification loss during training is very noisey and increasing. One possible reason for this is the way I have defined the computation of classification accuracy. if any of the following cases ocurr when an image is passed to the model:
- The model does not detect an object in the image (no box drawn).
- The model detects more than one object in the image.
This image is no longer considered in the classification calculation, since these ocurrences are the fault of the model's box detection, thus the classification accuracy should not be penalised. However, as shown in Figure 4, there were only 465 valid classifications considered (out of a possible 600 test images), so it is possible that the abnormal classification loss was caused by multiple object-detections on the single-object image dataset. This would be problematic for the classification loss because in this case, at least one of the detections is likely to have a poor IOU. This hypothesis is supported by the poor mAP0.5:0.95 and average IOU values, which indicate that the box detection could be improved (see "Metrics of Interest" section).

Another reason for this behaviour could be because the default training process has an IOU threshold of 0.2, which means that the model is sometimes attempting to perform classification when only a fraction of the object is inside the bounding box.

### Solution 1
The first stage of improvement will be specifying a different hyperparameter file, which should hopefully give improvement over all metrics/losses. By default, augmentations and hyperparameters are specified in yolov5_LC/data/hyps/hyp.scratch-low.yaml. This is a low-augmentation configuration designed for training the COCO dataset from scratch. It is possible that ample improvement could be achieved by simply changing this to the yolov5_LC/data/hyps/hyp.VOC.yaml hyperparameter file. This file is designed for training with a pretrained net, on the VOC dataset, which is more suitable since the VOC dataset is closer in size to the ISIC dataset, and we are using a pretrained net for this project. This is achieved by changing the train execution command in train.py from:

```python
os.system(f"python3 yolov5_LC/train.py --img 640 --batch -1 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5{yolo_model}.pt")
```
To:

```python
os.system(f"python3 yolov5_LC/train.py --img 640 --batch -1 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5{yolo_model}.pt --hyp yolov5_LC/data/hyps/hyp.VOC.yaml")
```

### Solution 2
Findings in the previous section seem to indicate that improving the bounding box detection will improve the classification loss, average IOU, and mAP. 
The first stage of improvement is based on YOLOv5 documentation found at (https://docs.ultralytics.com/tutorials/training-tips-best-results/) -> as mentioned in the previous section, the dataset doesn't seem to be big enough.YOLOv5 already implements data augmentation by default, however, the albumentations library can be used specify even more augmentations, which should improve the performance of the model for the ISIC dataset.


The documentation also mentions that turning down the gain for individual loss terms can also assist reduction of overfitment - so in this case, we will reduce the classification loss gain.

Increasing the IOU threshold when training has also been considered, however, it will be left at 0.2 for now, because the low threshold may promote more robust classification.



- more augmentation with albumentation lib - reference jocher saying that the set should have >10000 images
- research modifications to hyperparameters - probably not a good idea as they have realistically already been optimised
- weight initialisment
- Population Based Bandits (PB2)?

