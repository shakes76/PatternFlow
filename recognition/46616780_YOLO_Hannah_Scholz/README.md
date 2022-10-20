## YOLOV5 Image Detection
#### By Hannah Scholz - s4661678
This is a python based package that utilises a YOLOV5 model to detect skin lesions from the ISIC dataset.

----------

### Description of the algorithm
YOLO - "You Only Look Once", is a real-time object detection system which only looks once at an image to predict what 
objects are present and where they are located.
YOLO uses a single Convolutional Neural Network to detect objects in an image. 
The neural network divides the image into regions, predicts bounding boxes and calculates probabilities for each region.
The bounding boxes are weighted by the predicted probabilities. 
Compared to the original YOLO, YOLOV5 has a few changes to improve training and increase performance.
I chose the YOLOV5 model as it is very easy to use and is based in the PyTorch framework, something that we have been
utilising for most of the semester. 

### Problem it solves
In this specific circumstance with the dataset provided, early detection of problematic skin lesions could be one of the 
best preventative methods for stopping the development of life-threatening skin cancers. 
This model provides a solution to this problem. Using convolutional neural networks, the YOLOV5 model provides the 
ability for skin lesions to be identified using image detection. This could potentially aid medical professionals in 
correctly identifying cancerous skin lesions.

### How it works
YOLOV5 is a single stage object detector that has three main components:
1. **The model backbone** - A convolutional neural network that extracts important features from a given input image.
Networks. 

2. **The model neck** - Which generates a series of layers, which help the model to mix and combine features. 

3. **The model head** - Used to perform the final detection by applying anchor boxes, generates class probabilities
and bounding boxes.

The **activation function** used by YOLOV5 is the LeakyReLU and the sigmoid activation functions.  

The **optimisation function** used by YOLOV5 is either the stochastic gradient descent (SGD) function or the adam
optimiser. In this report the default optimiser used is SGD.

YOLOV5 also applies **data augmentation** during training, to increase the amount of data to improve the performance. 
This is necessary as the recommended number of images per class for YOLOV5 is over 1,500 images.

<img width="600" alt="Augmentation Visualisation" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/DataAugmentationVisualisation.jpeg">

### Model Architecture Visualisation
Here we can see a visualisation of the model parts in detail:

<img alt="ModelVisualisation" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ModelVisualisation.jpg">

---------

### Dependencies
#### Versions
Python - 3.9.12 (Was run in PyCharm 17.0.3)

Google Collab

#### How to run
1. Download images from: https://challenge.isic-archive.com/data/#2017
2. Put all images in specified folders - training, testing, validation, and create a labels folder to store the labels.
3. Run the dataset.py file, this will create the label files and preprocess the data.
4. Compress the data into a .zip file called Archive.zip which contains all the labels and images in the format below:

<img width="300" alt="FolderOrganisation" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/FolderOrganisation.jpg">

5. Download this .zip file to google drive
6. Run the YOLOV5_training.ipynb file - make sure the format of the files in folders is as specified above
7. Observe the results in the specific folders of YOLOV5, as noted in the .

### Example Inputs
<img width="300" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ExampleInput1.jpg"> <img width="300" alt="Example2" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ExampleInput2.jpg">

### Example Outputs
<img width="300" alt="Output1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_example1.jpg"> <img width="300" alt="Output2" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_example2.jpg">

---------------

### Data preprocessing
The data preprocessing is contained in the dataset.py file.
Since the training, testing and validation split is as given by the ISIC dataset, there is no requirement to determine 
how to split the data.
There was however many more aspects in which the data needed to be edited.
The images all had varying sizes for a start. YOLOV5 takes image sizes of 640x640, so all images were resized.
Next the dataset contained some unnecessary photos in a .png format that had to be removed. 
Bounding boxes also had to be determined for each mask photo:

<img width="200" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/ExampleMaskPhoto.png">
<img width="300" alt="Example1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/BoundingBoxOnMask.png">



Bounding boxes can be visualised using the predict.py file.
The .txt files that contain the coordinates of the bounding boxes and the class to which the image belongs too
is also created by running dataset.py file.

---------

### Results
The YOLOV5 has three main models available to train, a small, medium and large model. 

<img width="200" alt="Model Types" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotosForReadMe/YOLOv5Models.jpeg"> 

I first started training the YOLOV5 small model. Each model takes varying amounts of time to train and memory 
consumption. Since I am using Google Collaboratory to train the model I decided to start with the small model. Unfortunately 
Google Collaboratory only has a limited amount of memory available and CPU processing provided for free and so was 
difficult and a very long process to train the model without interruption. I am using the pretrained backbone of YOLOV5, 
the default hyperparameters, apart from the batch size which I have maximised as recommended, and I am using YOLOV5 
pretrained weights. 

---------------

### Yolov5s - small
As expected the YOLOV5 model increases in performance as the epochs increase, as an example different results from 
varying epoch numbers are displayed below. The aim is to reach an mAP score of 0.8 or higher. We also want to observe 
a good IOU score, which is the direct measurement of how close the predicted boxes match the label boxes.

#### 4 Epochs:
<img width="800" alt="4EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Results_Yolov5s_4Ep.png">

<img width="300" alt="Output1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Output1.jpg"> <img width="300" alt="Output2" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Output2.jpg"><img width="300" alt="Output3" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Output3.jpg">

#### 50 Epochs:
<img width="800" alt="50EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Results_Yolov5s_50Ep.png">

#### 100 Epochs:
<img width="800" alt="100EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Results_Yolov5s_100Ep.png">

#### 200 Epochs:
<img width="800" alt="200EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Results_YOLOV5_200Ep.png">

#### 300 Epochs:
<img width="800" alt="300EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/Results_YOLOV5s_300Ep.png">

<img width="300" alt="Output1" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_example1.jpg"> <img width="300" alt="Output2" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_example2.jpg"><img width="300" alt="Output3" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_example3.jpg"><img width="300" alt="Output4" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_example4.jpg"><img width="300" alt="Output5" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_example5.jpg">

As we can see from these results a metric/mAP_0.5 and metric/mAP_0.5:0.95 are slowly getting better and better, although 
do not seem to be increasing any further after 300 epochs. Using the function I have created to calculate the IOU metric
in the YOLOV5_training.ipynb file, the 300 epoch training model has an IOU value of 0.8643. This is perfect, although the 
mAP scores I believe could be better, as it is sitting between 0.6 and 0.7.

Next is to train using the medium YOLOv5 model instead to see if the performance improves further.

-----------
### Yolov5m - medium 
#### 4 Epochs:
<img width="800" alt="4EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/YOLOV5m_4Ep.png">

#### 50 Epochs:
<img width="800" alt="50EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/YOLOV5m_50Ep.png">

#### 100 Epochs:
<img width="800" alt="50EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/YOLOV5m_100Ep.png">

#### 300 Epochs:
<img width="800" alt="300EpochResult" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/YOLOV5m_300Ep.png">

<img width="300" alt="Example1_300Ep_M" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_Example1_M.jpg"> <img width="300" alt="Example2_300Ep_M" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_Example2_M.jpg"><img width="300" alt="Example3_300Ep_M" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_Example3_M.jpg"><img width="300" alt="Example4_300Ep_M" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_Example4_M.jpg"><img width="300" alt="Example5_300Ep_M" src="https://github.com/hannahscholz/PatternFlowHS/blob/topic-recognition/recognition/46616780_YOLO_Hannah_Scholz/PhotoResults/300Epoch_Example5_M.jpg">

As we can see from these results a metric/mAP_0.5 and metric/mAP_0.5:0.95 are slowly increasing through the increase in
epoch numbers. This is very similar to the small YOLOV5 model. Again it does not seem to be increasing any further 
after 300 epochs. Using the function I have created to calculate the IOU metric in the YOLOV5_training.ipynb file, 
the 300 epoch results have an IOU value of 0.8961.

--------

### Conclusion of results:
Through these results we can see the model receives high results for metric/mAP_0.5, metric/mAP_0.5:0.95 and IOU values.
Although we should note that the classification loss is increasing for both models. From research, I believe
this may be from the fact that the data is on the smaller side, as YOLOV5 creators recommend a minimum of 1,500 images 
per class. The ISIC dataset has significantly less than the recommended. 
Overall I have thoroughly enjoyed working on the YOLOV5 model!

---------

### File Contents:
dataset.py ---> Contains functions to preprocess the data.

modules.py ---> Since YOLOV5 can be followed model wise from the YOLOV5 GitHub this file is empty, a link
to the repository I used is linked above as a submodule.

predict.py ---> Contains code to draw bounding boxes on images, the YOLOV5_training.ipynb contains 
functions and code to display the images and results.

train.py ---> The YOLOV5_training.ipynb file contains everything this file should contain, the file was not
renamed as it would delete the commit history from GoogleCollab.

YOLOV5_training.ipynb ---> Contains the training of the model, shows results and calculates IOU values.

PhotoResults folder ---> Contains photos of the results used in the README.md

PhotosForReadMe folder ---> Contains photos used in the README.md


----------

### References:
https://github.com/ultralytics/yolov5

https://docs.ultralytics.com/tutorials/training-tips-best-results/

https://blog.roboflow.com/yolov5-improvements-and-evaluation/

https://arxiv.org/pdf/1506.02640.pdf

https://medium.com/mlearning-ai/training-yolov5-custom-dataset-with-ease-e4f6272148ad

https://github.com/ultralytics/yolov5/issues/388

https://github.com/ultralytics/yolov5/issues/36
