# Object Detection with YOLO in the ISIC Dataset
Please note that module.py is empty because the model itself is inside of the yolov5_LC git submodule.
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

Finally, the albumentations library, which is used to increase the augmentations, can be installed:

```linux
pip install -U albumentations
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
and select mode 0 from the options to download/arrange/preprocess the data completely. This will also print some outputs and save some images to misc_tests in order to validate correct function of the repo. 

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
The yolov5 github specifies the quantitative differences between the models, but basically, as the model gets larger, it will become slower, but perform better. See this link for specifications: https://github.com/ultralytics/yolov5#pretrained-checkpoints. It is reccomended to use the medium size model, as this is the model investigated in this report.

### Inference
In order to deploy the trained model, predict.py is available. predict.py contains the Predictor class, and a Predictor_Example_Use() function, which shows the user how to load the model, and produce and visualise object detection/classification on a single image. It also shows the user how to visualise comparisons between predicted and labelled object detection/classification and how to use utils_lib to compute IOU and find if the classification is correct (if the user wishes to run comparisons on labelled sets). To run this function simply enter the following command in terminal:

```linux
python3 predict.py
```
This will save any outputs to the pred_out folder.

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
### Trial 1
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

This seems to indicate that improving the bounding box detection will improve the classification loss, average IOU, and mAP. 

Another reason for this behaviour could be because the default training process has an IOU threshold of 0.2, which means that the model is sometimes attempting to perform classification when only a fraction of the object is inside the bounding box.

### Trial 2
The first stage of improvement will be specifying a different hyperparameter file, which should hopefully give improvement over all metrics/losses. By default, augmentations and hyperparameters are specified in yolov5_LC/data/hyps/hyp.scratch-low.yaml. This is a low-augmentation configuration designed for training the COCO dataset from scratch. It is possible that ample improvement could be achieved by simply changing this to the yolov5_LC/data/hyps/hyp.VOC.yaml hyperparameter file. This file is designed for training with a pretrained net, on the VOC dataset, which is more suitable since the VOC dataset is closer in size to the ISIC dataset, and we are using a pretrained net for this project. This is achieved by changing the train execution command in train.py from:

```python
os.system(f"python3 yolov5_LC/train.py --img 640 --batch -1 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5{yolo_model}.pt")
```
To:

```python
os.system(f"python3 yolov5_LC/train.py --img 640 --batch -1 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5{yolo_model}.pt --hyp yolov5_LC/data/hyps/hyp.VOC.yaml")
```

Increasing the IOU threshold when training has also been considered, however, it will be left at 0.2 for now, because the low threshold may promote more robust classification.

The model was trained for 350 epochs, but triggered early stoppage at 227 epochs (early stoppage is triggered when there is no improvement for 100 epochs). The training data is shown below:

Figure 5:

![image](https://user-images.githubusercontent.com/32262943/195963353-6d2942ee-6bd5-4780-8c8e-aca04f9ffffc.png)

The following observations can be drawn from this:
- The box loss improved significantly -> unlike the first trial, there is no spike in loss towards the end, and the final loss value has approximately halved that of the first trial.
- The object loss begins to overfits at ~75 epochs, which is earlier than the first trial, however, it does seem to achieve a slightly better loss value than the first trial.
- The classification loss is still very noisey, however it does appear to be an improvement over the first trial -> it no longer increases for the entire training period (begins to overfit at ~75 epochs), and is of a much lower value.

The model is run on the test dataset:

Figure 6:

![image](https://user-images.githubusercontent.com/32262943/195964807-c18c8b92-756a-43f8-bbe8-0f2f28daed96.png)

Figure 7:

![image](https://user-images.githubusercontent.com/32262943/195964841-88fe390f-0138-4e01-9f9f-4f77ea920920.png)

The following observations can be drawn from this:
- An increase in mAP0.5-0.95 is evident, which indicates improved bounding boxes, however, this value is still below the yolov5m listed value of 45.2.
- The improved mAP is shown in a marginal increase of average IOU (0.76 compared to 0.75), and a significant increase in valid classification boxes (484 compared to 465). An increase in valid classifications implies that the model is predicting more accurate bounding boxes (more boxes passing the IOU threshold), and not prediciting more than one box when it shouldn't. There was a slight increase (of 5) in lesions that were missed by the model, however, this is outweighed by the improvements in IOU and valid classifications.
- A significant decrease in classification accuracy (0.83 compared to 0.87) is observed, however this is still above the acceptable accuracy, and might be because the extra valid classifications observed in this trial are closer to the 0.5 IOU threshold. 

Another observation that was missed in trial 1 is the distribution of classes (melanoma cases) in the test set. As shown in Figure 6, there are 483 negative cases and only 117 positive cases, and the P, R, and mAP for positive cases is considerably lower. Thus, it is possible that the classification accuracy is only at an acceptable level because the model is guessing that all lesions are negative -> this would yield high accuracy if there aren't many valid positive lesions presented to the classifier. We can check this by graphing TP, TN, FP, FN predictions for the 484 valid classifications, and the 116 invalid detections:

Figure 8:

![image](https://user-images.githubusercontent.com/32262943/195989858-e9062c51-cb30-43a5-b6d3-d588e33876e0.png)

Figure 9:

![image](https://user-images.githubusercontent.com/32262943/195985351-1a69c317-a135-4f14-810f-ceb1a49806b3.png)

Here we can see that my hypothesis was quite accurate; the classification accuracy is skewed by the fact that there are many more negative melanoma cases in the dataset. In fact, if we calculate the precision and recall (from Figure 8) in regards to detection of positive melanoma cases:

P = TP/(TP+FP) = 50/50+43 = 0.538 -> "~54% of positive predictions are correct"

R = TP/TP+FN = 50/50+37 = 0.575   -> "~58% of positive cases are diagnosed successfully"

We can see that neither of these are particularly impressive, and certainly wouldn't be accepted in practice. This shows how much the classification accuracy has been skewed by the distribution of the training set. Furthermore, from Figures 8 and 9, we can see that ~24% of positive cases were unable to be detected/were incorrectly detected/had a very low IOU, while 18% of negative cases suffered this case. This makes sense because, due to the significantly larger number of negative cases in all datasets, the model would have much better detection of negative cases.

### Trial 3
This stage of improvement is based on YOLOv5 documentation found at (https://docs.ultralytics.com/tutorials/training-tips-best-results/) -> as mentioned in the previous section, the dataset doesn't seem to be big enough, especially for the positive cases. YOLOv5 already implements data augmentation by default, however, the hyperparemeter file used in trial 2 has some options to increase augmentations. In order to take advantage of augmentation settings that have been somewhat optimised by the yolov5 developers, we will combine the high-augmentation parameter settings from the COCO config file, with the other hyperparameter settings from the VOC hyperparameter file config (used in prev. trail). Some of the augmentation options were set to zero or very low, thus, the probabilities and magnitudes of augmentations such as rotation, translation, shear, perspective, and up/down flip ocurring were all increased.

Furthermore, YOLOv5 is integrated with the albumentations library, which means that once the albumentations dependency is installed, the albumentations class in the utils_2 folder can be modified to produce even more augmentations. This class was modified to introduce extra augmentations such as; blur, median blur, and to grey. 

After these modifications, there are more different augmentations, occurring more often, at higher magnitudes - with the aim of making a more robust YOLO model, which has much more data to train on. The training results are shown below:

Figure 10:

![image](https://user-images.githubusercontent.com/32262943/195987560-146e0a9f-ba0f-4671-900e-6d4d1f1bf56a.png)

The following observations can be made:
- The box loss is approximately the same as the previous trial.
- The object loss has a similar minimum, but doesn't overfit like the previous trial, it plateaus.
- There is a huge improvement in the classification loss, the noise is greatly reduced (even the training loss looks better), and it doesn't overfit at all, unlike the previous trial.
- The mAP plots also have reduced noise.

These observations imply that the increased augmentations have resulted in more stable training, with significantly reduced overfitting in the classification and object loss metrics. The test results are shown below:

Figure 11:

![image](https://user-images.githubusercontent.com/32262943/195988195-e656b631-377a-4b22-aff3-5d2179419de3.png)

Figure 12:

![image](https://user-images.githubusercontent.com/32262943/195988209-666a1d8d-eb99-4525-9b62-74c14ca4d508.png)

From these results we observe that:
- The mAP has increased significantly, now above the specified yolov5m performance (45.2) with a value of 46.6.
- The IOU has reached an acceptable level of 0.81, however, there is a decrease in valid object detections.
- The classification accuracy has increased to ~0.87, with an increase in the number of valid classifications.

At this point, it can definitely be said that the augmentations have improved the performance, however, we must also check the distributions of valid and invalid classifications:

Figure 13:

![image](https://user-images.githubusercontent.com/32262943/195990029-fa531bcf-ed15-495c-a377-f1aafd1fc57b.png)

Figure 14:

![image](https://user-images.githubusercontent.com/32262943/195988544-c5f47511-4257-4756-840b-275fb0c7cc5a.png)

From this, we can see that the problem clearly has not been solved. Even though the average IOU and classification accuracy has improved, we can see that this is directly correlated with a reduction in valid positive bounding boxes (~29% invalid), and increase in valid negative boxes (~15% invalid), so it would appear that, instead of learning how to classify positive cases better, the model has just taken advantage of the distribution. This can be further investigated by computing the precision and recall in regards to detection of positive melanoma cases:

P = TP/(TP+FP) = 39/39+21 -> "~65% of positive predictions are correct"

R = TP/(TP+FN) = 39/39+44   -> "~46% of positive cases are diagnosed successfully"

while there is a significant increase in precision, there is also a significant decrease in recall. In the case of medical diagnosis, recall is usually more important, because it shows how many positive cases were missed.

## Conclusion
### Summary
Clearly, despite the third trial achieving the targetted benchmarks (45.2 mAP05:0.95, 0.8 IOU, 0.8 classification accuracy), this model is far from successful, with the final trial only successfully diagnosing 46% of positive melanoma cases in the ISIC 2017 dataset. Furthermore, this metric doesn't consider the cases which were missed due to poor bounding box detection, so it is actually much less than the computed recall metric reports.

It is hypothesised that this is mostly a dataset issue. More specifically, the distribution of positive and negative cases is not favourable, (as shown in Figure 2), with 1600 negative cases, but only 400 positive cases in the training set, and similar distributions in the validation and test sets. This is supported by the significant improvements in classification/object loss and mAP, which resulted from an increase in augmentation frequency, magnitude, and variety. It is also supported by documentation which states that the dataset should have at least 10000 labelled objects per class (https://docs.ultralytics.com/tutorials/training-tips-best-results/).

### Recommendations
With augmentation applied, the model was able to detect and classify negative melanoma cases quite well, and as such the recommendations for future work could involve one or more of the following:
- Combine the positive data with positive data from some of the other ISIC datasets, this way, the model will benefit from increased negative cases, and a more favourable distribution.
- Decrease augmentation for negative cases and increase augmentation for positive cases. This will have the same affect as the above suggestion, albeit with less impact.
- Remove some of the positive data to make the distribution closer to even, then increase augmentations to make up for the decrease in dataset size.

Some other alterations that could be investigated are as follows:
- tune the IOU threshold for training - it is possible that the relatively low threshold of 0.2 is causing unstable learning.
- tune the gain of object, box, and classification loss - this can be used to tune which loss terms are more important to the model. Possibly an increase in box loss gain could improve all three losses as it may result in improved bounding boxes.
- A hyperparameter tuning algorithm, such as Population Based Training (PBT) or Population Based Bandits (PB2) could be used to optimise hyperparameters for this specific case, however, it is recommended that this particular option is not implemented until the model is closer to acceptable performance.

### Reproducability of Results
Assuming the use of the same ISIC 2017 dataset, these results should be reproducable, as long as usage instructions are followed, with the correct yolov5 size model (m) used, and the correct number of epochs. The exception to this is if a different machine is used, it may not be able to achieve the same batch size (my machine used a batch size of 21, using yolov5 auto size functionality). If the batch size is too small, this will produce unfavourable batch normalisation results.

Deployment of the trained model on new data is not adviseable, due to the size of the training dataset (as discussed previously), and the unimpressive Recall and Precision results.

## Some Example Outputs of the Third Model:
Labels-batch0:

![image](https://user-images.githubusercontent.com/32262943/195992119-dabe9995-f5b4-47d0-879d-25412772d66e.png)

Predictions-batch0:

![image](https://user-images.githubusercontent.com/32262943/195992148-19c27dc4-a2cd-42b6-87e9-761fa3d6722e.png)

Labels-batch1:

![image](https://user-images.githubusercontent.com/32262943/195992178-e49bff1f-4923-4bb3-9aa9-df34691c8bc5.png)

Predictions-batch1:

![image](https://user-images.githubusercontent.com/32262943/195992198-b0cb0487-d347-49c6-867e-3b03233a4616.png)


