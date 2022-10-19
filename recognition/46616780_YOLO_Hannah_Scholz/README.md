
## YOLOV5 Image Detection
#### By Hannah Scholz - s4661678
This is a python based package that utilises a YOLOV5 model to detect skin lesions.

### Description of the algorithm
YOLO - "You Only Look Once", is a real-time object detection system which only looks once at an image to predict what 
objects are present and where they are located.
YOLO uses a single Convolutional Neural Network to detect objects in a full image. 
The neural network divides the image into regions, predicts bounding boxes and calculates probabilities for each region.
The bounding boxes are weighted by the predicted probabilities. 
Compared to the original YOLO, YOLOV5 has a few changes to improve training and increase performance.
I chose the YOLOV5 model as it is very easy to use and is based in the PyTorch framework, something that we have been
utilising for most of the semester. 

### Problem it solves:
In this specific circumstance with the dataset provided, detecting problematic skin lesions early could be one of the 
best preventative methods for stopping the development of life-threatening skin cancers.
This model provides a solution to this problem. Using convolutional neural networks, the YOLOV5 provides the ability 
for skin lesions to be identified using image detection. This could potentially aid medical professionals to correctly 
classify cancerous skin lesions.

### How it works:
YOLOV5 is a single stage object detector that has three main components:
1. **The model backbone** - A convolutional neural network that extracts important features from a  given input image.
Networks. 

2. **The model neck** - Generates a series of layers, which help the model to mix and combine features. 

3. **The model head** - Used to perform the final detection by applying anchor boxes, generates class probabilities
and bounding boxes.

The **activation function** used by YOLOV5 is the LeakyReLU and the sigmoid activation function.

The **optimisation function** used by YOLOV5 is either the stochastic gradient descent function or the adam
optimiser. 

YOLOV5 also applies **data augmentation** during training, to increase the amount of data to improve the performance. This
is necessary as the recommended number of images per class for YOLOV5 is over 1,500 images.

<img width="400" alt="Augmentation Visualisation" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/DataAugmentationVisualisation.jpeg">

### Model Architecture Visualisation:
Here we can see a visualisation of the model parts:


<img width="600" alt="ModelVisualisation" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ModelVisualisation.jpg">

### Dependencies:
#### Versions:
MacOS Monterey 12.6

Python - 3.9.12

Google Collab

#### How to run:
1. Download images from: https://challenge.isic-archive.com/data/#2017
2. Put all files in specified folders - training, testing, validation 
3. Run the dataset.py file
4. Compress the data into a .zip file called Archive.zip
5. Download this file to google drive
6. Run the train.ipynb file - make sure the format of the files in folders is as specified below:

<img width="300" alt="FolderOrganisation" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/FolderOrganisation.jpg">

7. Observe the results using train.ipynb file.

### Example Inputs:
<img width="200" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ExampleInput1.jpg">

<img width="200" alt="Example2" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ExampleInput2.jpg">

### Example Outputs:




### Data preprocessing:
The data preprocessing is contained in the dataset.py file.
Since the training, testing and validation split is as given by the ISIC dataset, there is no requirement to determine 
how to split the data.
There was however many more aspects in which the data needed preprocessing.
The images all had varying sizes for a start. YOLOV5 takes image sizes of 640x640, so all images were resized.
Next the dataset contained some unnecessary photos in a .png format that had to be removed. 
Bounding boxes had to be determined for each mask photo:
<img width="200" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ExampleMaskPhoto.png">

<img width="200" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/BoundingBoxOnMask.png">


### Results:
### Yolov5s - small
#### 4 Epochs:
<img width="600" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/Results_Yolov5s_4Ep.png">


#### 50 Epochs:
<img width="600" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/Results_Yolov5s_50Ep.png">


#### 100 Epochs:
<img width="600" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/Results_Yolov5s_100Ep.png">


### Yolov5m - medium 


### References:
https://github.com/ultralytics/yolov5
https://docs.ultralytics.com/tutorials/training-tips-best-results/
https://blog.roboflow.com/yolov5-improvements-and-evaluation/
https://arxiv.org/pdf/1506.02640.pdf
https://medium.com/mlearning-ai/training-yolov5-custom-dataset-with-ease-e4f6272148ad


