# Yolov3 ISIC dataset lesion detection

<br>
The skin is the largest organ of the human body. Some skin diseases, such as malignant black, melanoma. These diseases are likely to cause death, and the consequences of misdiagnosis will be very serious. Moles are skin diseases with mild symptoms, but some fatal skin diseases such as malignant black and mildly symptomatic skin diseases and their Similar. Effectively locating and detecting these skin diseases plays a critical role in saving the lives of patients. The ISIC data set contains 23k melanoma examples of malignant and benign images that classify skin injuries. Yolo is a kind of slider image The detection technology is different from mask-rcnn in that it uses a one-stage algorithm, which only uses a convolutional neural network to directly predict the classes and positions of different targets. In here, YoloV3 is used to detect the lesion area of the ISIC data set. The goal of this task is that all detections having a minimum Intersection Over Union of 0.8 on the test set.
<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/65603393/139587701-31aa3510-b313-4c36-a728-b9fef504524a.png" width="650" height="500">
  <br>
  <i>Figure 1: ISIC_0016028.jpg detection example.</i>
 </p>
<br>
<br>

* [ISIC dataset](#ISIC-dataset)<br>
* [YoloV3](#YoloV3)<br>
* [Run&use](#Run&use)<br>
* [Results](#Results)<br>
* [Dependencies](#Dependencies)<br>
* [Reference](#Reference)<br>


## ISIC-dataset
In this task, we have given the preprocessed data set, which includes 2594 pictures and the corresponding segmented black and white pictures. 
<p align="center">
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/138678240-f794ef8b-b534-4a91-a953-96f2cb366411.jpg" width="650" height="500">
  <br>
  <i>Figure 2: Training image.</i>
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/138678838-ce003a09-fb42-41b2-96dd-31819110ff42.png" width="650" height="500">
  <br>
  <i>Figure 3: Segmented image.</i>
</p>

## YoloV3
Yolov3 is based on the changes made in v2 and v1. The main improvements are: 
1. Adjusted the network structure to use multiscale features for object detection.
2. Object classification replaced softmax with logistic.
<p align="center">
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/138707537-3a35da27-fb4a-49a5-ab36-e0b259372e13.png" width="300" height="250">
  <br>
  <i>Figure 4: Resnet skip connection.</i>
</p>
YoloV3 adopts Darknet53 for image feature extraction. In Darknet53, some convolutional layers adopt the practice of residual network, and some layers directly set up skip connections to make the network deeper and extract more features. The pooling layer is replaced by a convolution operation with a step size of 2 for deep extrating the features.
<h3>Darknet53</h3>
<p align="center">
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/138707097-e2e51e03-f852-4be5-80e9-740be4a0f3a9.png" width="650" height="580">
  <br>
  <i>Figure 5: Darknet53 and upsampling.</i>
</p>
YoloV3 uses three different scales of feature maps for target detection to solve the problem that YoloV2 and V3 are insensitive to small targets, which are 1313, 26x26, and 52x52, respectively, to detect targets of three different sizes: large, medium, and small.
The feature fusion layer selects the three scales of feature maps produced by DarkNet as input, and fuses the feature maps of each scale through a series of convolution layers and upsampling.

YOLOv3 divides the input image into SxS lattices, and each lattice predicts B bounding boxes, each bounding box prediction includes: x, y, width, height, Confidence and the probability of C categories, so the number of channels in the output layer of YOLOv3 is Bx(5 + C), in here for the ISIC dataset that is 3x(5+1)=18, 3 means a grid cell contains 3 bounding boxes, 4 means the 4 coordinate information of the box, and 1 means Confidence score. 

## Run&use
### Pre-processing
In the data pre-processing stage, the ISIC dataset segmentation image set is used. The first white pixel values (x_min,y_min) and (x_max,y_max) of each image were found during image pixel iterations and making to tags for the xml files. Also, when splitting the dataset, the training and test sets split into 80% and 20%. According to the YoloV3 data loading method, these training and test sets are made into absolute paths and stored in train.txt and test.txt for Dataloader.
Finally, in the process of loading the dataset in pytorch, when the imported data is the training set, the images will be enhanced to reduce the impact of insufficient data, improving the robustness of the model, providing various "invariants" to the model, and to increase the model's ability to resist overfitting, including resizing, adding gray bars, and image flipping. 
For making the xml file for training, you need to prepare the segmented ISIC dataset image and put it in the dataset/Mask_labeling folder. And run

    python dataset/Annotations/making_xml.py

After completing the xml file production, you need to split the test set and training set of the data set, here. The initial segmentation is set to 8:2, you can enter making_annotation.py to make custom adjustments. And run

    python making_annotation.py

### Training
If you want to start training the network again. Need to set weights_path in drive.py to "" and run train.py at the same time.
If you want to load, use pre-trained weights to make predictions. You need to click here to download the weights and place the weights in the results/weights/ directory.

    python train.py and edit driver.py


### Predicting
After you have the weights, if you want to predict the picture, you need to correct the img_name to the file name for your need. And run

    python driver.py or test.py


## Results
In the training process, the segmented training method is used to better visualize the way the loss decreases in different stages. At the same time, the rate of IOU on the test set exceeds 0.8 for the first time at 180 iterations, which is 0.835.

<p align="center">
  
  <img src="https://user-images.githubusercontent.com/65603393/139538254-ca583fa7-c163-4ff1-8e68-976c028614bb.png" width="650" height="380">
  <br>
  <i>Figure 6: ISIC dataset first 9 epochs.</i>
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/139538340-83ee43cd-1e61-45f9-92a6-9bb96b90ca1b.png" width="650" height="380">
  <br>
  <i>Figure 7: ISIC dataset epoch 10-40</i>
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/139538438-f970684f-efc1-44ed-b161-b99ed08df998.png" width="650" height="380">
  <br>
  <i>Figure 8: ISIC dataset epoch 41-180.</i>
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/139584365-ce215657-c0ee-44d4-a4e8-4939b0c27ed8.png" width="650" height="380">
  <br>
  <i>Figure 9: ISIC dataset test set IOU rate.</i>
  <br>
  <img src="https://user-images.githubusercontent.com/65603393/139586796-b0c597f0-3aff-4fb9-90f0-5744f8d476a7.png" width="650" height="380">
  <br>
  <i>Figure 10: ISIC dataset test set IOU in particular epochs.</i>
  <br>
</p>


## Dependencies
* Python 3.7
* Numpy 1.20.3
* Torch 1.9.0
* Torchvision 0.4.0
* Opencv-Python 4.5.3.56
* PIL 1.1.7
* Matplotlib 3.1.2
## Reference
GitHub - bubbliiiing/yolo3-pytorch. (2020, September 9). GitHub. Retrieved October 15, 2020, from https://github.com/bubbliiiing/yolo3-pytorch
<br>
ISIC Challenge. (n.d.). ISIC Challenge. Retrieved October 2, 2021, from https://challenge.isic-archive.com/
<br>
Detailed explanation of YOLOv3 loss function. (n.d.). Darknet53. Retrieved October 14, 2021, from https://www.fatalerrors.org/a/detailed-explanation-of-yolov3-loss-function.html
<br>
Adaloglou, N. (2020, March 23). Intuitive Explanation of Skip Connections in Deep Learning. AI Summer. Retrieved October 14, 2021, from https://theaisummer.com/skip-connections/
