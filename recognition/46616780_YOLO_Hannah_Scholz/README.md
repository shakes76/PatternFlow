
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


### Example Outputs:


### Data preprocessing:
Justify your training, validation and testing splits of the data.
Training, testing and validation split is as given by the ISIC dataset.
Image dimensions --> 640x640 as this is what dimension YOLOv5 was designed for.

### Results:

### References:
https://pjreddie.com/darknet/yolo/
https://arxiv.org/pdf/1506.02640.pdf
https://medium.com/mlearning-ai/training-yolov5-custom-dataset-with-ease-e4f6272148ad


