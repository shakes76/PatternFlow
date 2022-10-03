# Object Detection with YOLO in the ISIC Dataset
## YOLO:
This section will detail What the algorthm is, what problem it solves, and how it works.
### Description of YOLO
The YOLO model family consists of multiple different versions of an object detection model. Object detection models aim to search for object classes within given images. Once found, these objects are indicated by a bounding box, and classified with a label. YOLO stands for You Only Look Once, because each frame (or image) is only passed through the model once. The result of this is that YOLO models are smalller and faster  than other object detection implementations, which traditionally passed the image through once for defining bounding boxes, and then again to classify the box classes. (https://blog.roboflow.com/a-thorough-breakdown-of-yolov4/)

### YOLO Applications
YOLO models excel in realtime object detection, thanks to their but fast performance. These models are also very lightweight, which not only means that they can be implemented on video feeds at relatively high frame rate, they can also be deployed on native hardware easier than other models, because they do not require as much computing power. This project aims to use a YOLO model to detect and classify lesions within the ISIC dataset. (https://blog.roboflow.com/a-thorough-breakdown-of-yolov4/)

### A Word on the ISIC dataset.
ISIC is an organisation that aims to "Support efforts to reduce melanoma-related deaths and unnecessary biopsies by improving the accuracy and efficiency of melanoma early detection." The organisation aims to do this by engaging computer science communities to improve diagnostic accuracy, with the aid of AI techniques. (https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/aboutIsicGoals)

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

### YOLOv__ netowrk model: (select a yolo pls)

