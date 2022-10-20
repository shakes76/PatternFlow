# Lesion Detection with YOLOv5
This package utilizes [YOLOv5](https://github.com/ultralytics/yolov5) to train a model to detect lesions within various different images. 

## Overview

### Background



### YOLOv5 Model Architecture



### Data
For this task, the [ISIC 2017 Dataset](https://challenge.isic-archive.com/data/#2017) was used for training. The dataset contains 2000 images of lesions that are either melanoma or benign. YOLOv5 requires a specific format - that is it needs the images as well as labels. For this process, [Roboflow](https://roboflow.com/) was used as it simplifies the process of labelling the images. All the images were labelled and then exported to YOLOv5 PyTorch format. Since it would take a long time to label all 2000 images, around 300 images were selected and it was 90%, 5%, 5% split. Due to the fact that there are only 300 images, 90% was chosen so that there are more used in the training process. The results of the data preprocessing can be found in the Data folder.


### Results
The following two images show examples of inference using the model. It can be seen that it does it with 80% accuracy. The model is able to distinguish between benign and melanoma.  
<p align="center"><img src="./Results/ISIC_0000148.jpg" width="800"></p>
<p align="center"><img src="./Results/ISIC_0000020.jpg" width="800"></p>
The following image shows the results from the training:
<p align="center"><img src="./Results/results.png" width="800"></p>
<p align="center"><img src="./Results/model_summary.png" width="800"></p>

## Installation
To use this repo, clone it and make sure all the dependencies are installed

### Dependencies 
- Python = 3.10.2
- PyTorch = latest version
- Pillow = latest version
- Git = latest version
- [YOLOv5](https://github.com/ultralytics/yolov5)


## Usage
Modules.py contains an interface for downloading YOLOv5 from GitHub. If it does not work then clone the YOLOv5 repo into the project folder. 


## References
