# Lesion Detection with YoloV1
This is a python based package that utilizes a custom [YoloV1](https://arxiv.org/abs/1506.02640v5) model to detect skin lesions.
## Description
#### Background 
Australia has one of the highest rates of deaths by skin cancer in the world[1]. Detecting problematic skin lesions early is one of the best preventative methods for stopping the progression life threating cancer. The most common method in peforming this detecting is seeing a dermatologist[2], unfortunely this depends on the ability of dermatologist and the results can be varied[3]. 

#### Solution
This module gives a partial solution to this problem. Using the help of Convolutional Neural Networks, this module provides the ability for skin lesions to be detected using image detection, this can aid a dermatologist in their search for cancerous skin lesions. Further work will go into classifying these lesions. 

### Model Architecture:
This module utilizes a slight variation on the YoloV1 Architecture. 
<p align="center"><img src="./images/YOLOV1.png" width="600"></p>
The change? At each CNN layer, Batch Normalisation has been introduced as a way of speeding up the training process. 

### Loss Function:
As specified in the YoloV1 paper, the yolov1 uses a custom loss function. 
<p align="center"><img src="./images/yoloLoss.png" width="300"></p>
There is quite a bit to unpack here, so I suggest reading into it  [here](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088) 

### Metric:
The yolo paper uses the mean average precision (so called mAP) for its metric. This implementation instead uses the Jaccard Index (as known as 'IoU'). The jaccard index is calcaluted on the highest confidence bounding box. Below is visual representation of this equation:
<p align="center"><img src="./images/iou.PNG" width="600"></p>

### Dataset:
This model is trained on the [ISIC 2018](https://challenge2018.isic-archive.com/) Dataset. This dataset includes skin lesions and their respective ground truth segmentations. 
<p align="center"><img src="./images/ISIC.PNG" width="600"></p>

## Results:
This section is empty because no results were achieved :) 
## Usage:
The model.py contains the class YOLOV1 which contains all the neccessary information to train a new model or load existing weights. 

### Loading Provided Weights:
1) Create a new YoloV1 object:
```
from model import YoloV1
yolo = YoloV1()
```
2) Load the weights (checkpoint in repo)
```
yolo.loadWeights('./checkpoint')
```
3) Make a prediction
```
result = yolo.predictData(testSet)
```
### Training on new data
An example of training on the ISIC dataset is in the driver.py file, this covers and provides useful functions in data preprocessing for yolo. To train a new model:
1) First create a new YoloV1 object and declare the constants. 
    ```
    from model import YoloV1
    yolo = YoloV1(imageWidth=300, imageHeight=212, S=12, B=2, C=20)
    ```
    YoloV1 accepts 7 optional parameters, of which the first five are important and dependant on your data. They are as follows:
    - imageWidth: The width of all of your images. Noting that this number needs to be the same for every image in your dataset.
    - imageHeight: The height of all of your images. Noting that this number needs to be the same for every image in your dataset.
    - S: The number of cells to divide your image into. Divides your image into a S*S cell.
    - B: The number of bounding boxes to be predicted per cell. 
    - C: The number of classes that are contained in your dataset. This should match the number of classes in your bounding boxes. 
2) Compile your model, tuning any optional paramters. 
```
yolo.compile()
```
By default clipnorm is introduced at -1,1 too keep the network stable
3) Run your model and supply your data in the correct format:
```
yolo.runModel(training_data, validation_data, epochs=200)
```
The model accepts training and validation data in the form (image, groundTruth). Where:
- image is: (imageWidth, imageHeight, channels)
- groundTruth is: (S, S, 1, 5+num classes). Noting... that the groundTruth can only contain one true bounding box inserted at the correct S,S position. Every where else can be all zeros. 

4) Make a prediction:
```
result = yolo.predictData(testSet)
```

## References:
1. Sinclair, C. and Foley, P. (2009). Skin cancer prevention in Australia. British Journal of Dermatology, 161, pp.116–123.
2. Aitken, J.F., Janda, M., Lowe, J.B., Elwood, M., Ring, I.T., Youl, P.H. and Firman, D.W. (2004). Prevalence of Whole-Body Skin Self-Examination in a Population at High Risk for Skin Cancer (Australia). Cancer Causes & Control, 15(5), pp.453–463.
3. Jerant, A.F., Johnson, J.T., Sheridan, C.D. and Caffrey, T.J. (2000). Early Detection and Treatment of Skin Cancer. American Family Physician, [online] 62(2), pp.357–368. Available at: https://www.aafp.org/afp/2000/0715/p357.html?searchwebme [Accessed 9 Oct. 2020]