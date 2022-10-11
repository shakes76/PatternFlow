
## YOLO V5 Image Detection
#### By Hannah Scholz - s4661678
This is a python based package that utilises a YoloV5 model to detect skin lesions.

### Description of the algorithm
YOLO - "You Only Look Once", is a real-time object detection system which only looks once at an image to predict what 
objects are present and where they are located.
YOLO uses a single Convolutional Neural Network to detect objects in a full image. 
The neural network divides the image into regions, predicts bounding boxes and calculates probabilities for each region.
The bounding boxes are weighted by the predicted probabilities. 
Compared to the original YOLO, YOLOv5 has a few changes to improve training and increase performance.

### Problem it solves:
In this specific circumstance with the dataset provided, detecting problematic skin lesions early could be one of the 
best preventative methods for stopping the development of life-threatening skin cancers.
This model provides a solution to this problem. Using convolutional neural networks, the YOLOv5 provides the ability 
for skin lesions to be identified using image detection. This could potentially aid medical professionals to correctly 
classify cancerous skin lesions.

### How it works:


### Model Architecture Visualisation:


### Dependencies:


### Example Inputs:
Notes:
Do not need superpixel pictures
Do not need JSON files
Two classes: with and without melanoma

### Example Outputs:


### Data preprocessing:
Justify your training, validation and testing splits of the data.
Image dimensions --> 640x640 as this is what dimension YOLOv5 was designed for.

### Results:

### References:
https://pjreddie.com/media/files/papers/YOLOv3.pdf
https://pjreddie.com/darknet/yolo/
https://arxiv.org/pdf/1506.02640.pdf

https://github.com/edwardguil/PatternFlow/tree/ffad4e701f303b069689e053e97b52555a5041f8/recognition/4576111-YOLO-ISICs(2018)
https://github.com/lucaslin2020/PatternFlow/blob/d3918a651188ad6002771327106bc3a389c5fe71/recognition/s4569154/README.md
https://github.com/shakes76/PatternFlow/blob/b7fb6f55a8abafa4698c535276b560037181f232/recognition/Skin_Lesion_detection_using_YOLOv3/README.md
https://github.com/shakes76/PatternFlow/tree/a040943f9e3b11578ef99bda8fcbb7508791c4b6/recognition/S4607867_Jiaqiyu
https://github.com/shakes76/PatternFlow/blob/57672770f7f73517e88143d7ba69d286999ea60c/recognition/s4548700/README.md
