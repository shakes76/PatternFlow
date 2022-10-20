# Lesion Detection with YOLOv5
This package utilizes [YOLOv5](https://github.com/ultralytics/yolov5) to train a model to detect lesions within various different images. 

## Overview

### Background
In Australia, it is estimated that two out of three people will be diagnosed with skin cancer by the age of 70, and this amounts to nearly 2000 deaths due to skin cancer each year[1]. As such, Australia has one of the highest rates of skin cancer in the world[1]. Therefore, detecting skin cancer has never been more important, and the most common way right now is to go to a skin clinic and have a doctor look at it. The purpose of this repositry is to develop a model that can successfully detect if a lesion is benign or cancerous (melanoma).


### YOLOv5 Model Architecture
This repositry uses YOLOv5, which as of right now, does not have a paper. Therefore, it is hard to 


### Data
For this task, the [ISIC 2017 Dataset](https://challenge.isic-archive.com/data/#2017) was used for training. The dataset contains 2000 images of lesions that are either melanoma or benign. YOLOv5 requires a specific format - that is it needs the images as well as labels. For this process, [Roboflow](https://roboflow.com/) was used as it simplifies the process of labelling the images. All the images were labelled and then exported to YOLOv5 PyTorch format. Since it would take a long time to label all 2000 images, around 300 images were selected and it was 90%, 5%, 5% split. Due to the fact that there are only 300 images, 90% was chosen so that there are more used in the training process. The results of the data preprocessing can be found in the Data folder.


### Results
The following two images show examples of inference using the model. It can be seen that it does it with 80% accuracy. The model is able to distinguish between benign and melanoma.  
<p align="center"><img src="./Results/ISIC_0000148.jpg" width="800"></p>
<p align="center"><img src="./Results/ISIC_0000020.jpg" width="800"></p>
The following image shows the results from the training:
<p align="center"><img src="./Results/results.png" width="800"></p>
<p align="center"><img src="./Results/model_summary.png" width="800"></p>

It can be seen that the model has a mAP50 score of 0.833 which means the model is good at detecting if a lesion is benign or cancerous. However, the mAP50 score for Melanoma is 0.788 which is quite high. In some cases though when testing the model on an image it detected 

## Installation
To use this repo, clone the repositry. The repo has the following dependencies so make sure they are installed except YOLOv5.

### Dependencies 
- Python = 3.10.2
- PyTorch = latest version
- Pillow = latest version
- Git = latest version
- [YOLOv5](https://github.com/ultralytics/yolov5)


## Usage
Modules.py contains an interface for downloading YOLOv5 from GitHub. If it does not work then clone the YOLOv5 repo into the project folder. 

To perform inference:
```
from predict import *
model = load_model("path_to_best.pt")
predict(model, "path_to_image")
```
Using the above code will perform inference on the given image. A folder called 'runs' will be created and the image will be saved there.

To reproduce results clone this repositry and run the following script:
```
from train import *
from modules import *
from dataset import *

get_yolo() // install yolov5
install_requirements() // installs necessary requirements 
move_data_file()

train(16, 500, 24, "yolov5l.pt", "yolo_lesion_det")
```
If the train() function does not work run the following command in the terminal in the yolov5 directory:
```
python train.py --img 640 --batch 16 --epochs 500 --data data.yaml --weights yolov5l.pt --workers 24 --name yolo_lesion_det
```
To train on a new dataset, use Roboflow to label the images and then export to YOLOv5 PyTorch format. Rename the folder to Data. Within the folder, find data.yaml and change the first 3 lines to this:
```
train: ../Data/train/
test: ../Data/test/
val: ../Data/valid/
```
In a python script, have the following:
```
from train import *
from modules import *
from dataset import *

get_yolo() // install yolov5
install_requirements() // installs necessary requirements 
move_data_file()

train(batch_size, epochs, workers, weights, name)
```
If the train function does not work run the following in the terminal in the yolov5 directory:
```
python train.py --img image_size --batch batch_size --epochs epochs --data data.yaml --weights weights --workers workers --name name
```
Once the training has finished, the weights will be in yolov5/runs/name/weights/. These weights can then be used for inference. 

## References
1. Sunsmart, n.d., Skin cancer facts and stats, Retrieved From: https://www.sunsmart.com.au/skin-cancer/skin-cancer-facts-stats
