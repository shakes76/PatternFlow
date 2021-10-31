# The Detection and Identification of Yolov3 on ISICs Pictures
This project is the detections within the ISICs dataset with YOLOv3 network with all detections having a minimum Intersection Over Union of 0.8 on the test set and a suitable accuracy for classification.
## The principle of Yolov3
Yolov3 is based on a niche deep learning framework——darknet's target detection open source project. Darknet is short and powerful, fast, and gives full play to the parallel computing functions of multi-core processors and GPUs. Therefore, Yolov3’s rapid detection is just right for our projects that require real-time detection of video frames; in addition, its accuracy is also very high, and it has a very high accuracy rate on objects of medium to small size, thanks to its The training method (will be introduced below), but for large-sized objects, such as objects that account for 60% of the entire image, the recognition rate is not satisfactory.[1]
![image](https://user-images.githubusercontent.com/87461237/139569875-f9094ecc-2023-4fe9-a0f1-8d87aadff9bd.png)

The basic idea of this algorithm: First, extract features from the input features through a feature extraction network, and obtain a feature map output of a specific size. The input image is divided into 13×13 grid cells, and then if the center coordinates of an object in the real frame fall in a certain grid cell, then the grid cell will predict the object. Each object has a fixed number of bounding boxes. There are three bounding boxes in YOLO v3. Logistic regression is used to determine the regression box used for prediction.

## The network structure [5]
![139566717-231cc039-41ba-4eb3-9e45-6f0166ac62ac](https://user-images.githubusercontent.com/87461237/139570034-8206dd78-6fca-4ffa-bd70-896f950cc721.png)
**DBL**: Darknetconv2d_BN_Leaky in the code is the basic component of yolo_v3. It is convolution+BN+Leaky relu.

**resn**: n represents a number, res1, res2, …, res8, etc., indicating how many res_units are contained in this res_block.

**concat**: tensor splicing. Join the upsampling of the darknet middle layer and a later layer. The splicing operation is different from the residual layer add operation. The splicing will expand the dimension of the tensor, while the direct addition of add will not cause the tensor dimension to change.

YOLOv3 predicts an objectness score for each bounding box using logistic regression. This should be 1 if the bounding box prior overlaps a ground truth object by more than any other bounding box prior. If the bounding box prior is not the best but does overlap a ground truth object by more than some threshold we ignore the prediction, following [17]. We use the threshold of .5. Unlike [17] our system only assigns one bounding box prior for each ground truth object. If a bounding box prior is not assigned to a ground truth object it incurs no loss for coordinate or class predictions, only objectness.

## Description of usage and comments throughout scripts
1. python3.8
2. CUDA：a general purpose parallel computing platform and programming model that leverages the parallel compute engine in NVIDIA GPUs to solve many complex computational problems in a more efficient way than on a CPU.
3. OpenCV
4. os
5. darknet framework：Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
## Dataset
The training set is the training set of 2018ISICs, which contains 2518 original pictures and 2518 ground truths. The test set is the test set of 2017ISICs, including 600 original pictures and 600 ground truths each. This is the download address of my dataset: https://challenge.isic-archive.com/data/
## Model Training and Best Model Selection [2]
The Yolov3 model is trained on the ISIC training data set and the best model has been selected on the basis of improvement of validation data set dice loss. The specific steps are following:
1. Download 2018 ISICs trainning data from the url in the assessment pdf.
2. Transform the ground truth pictures to numerical tuples in txt and copy these txts into original images dataset.
![1635659857(1)](https://user-images.githubusercontent.com/87461237/139570164-eedb85be-0d8d-4eb6-9cdb-163b8b7bba38.png)
4. Download darknet framework and activate it.
5. Upload the latest dataset and extract their paths
6. Edit some of the config files.

![1635657068(1)](https://user-images.githubusercontent.com/87461237/139568833-832ad11c-84c3-49ee-8faa-2612b639be3d.png)
7. use the function in darknet to train the model according the trainning set.
![1635656968(1)](https://user-images.githubusercontent.com/87461237/139568778-cd823136-9e95-40ab-ac2c-7b1d47e4630a.png)
8. Tuning and validate each models
9. Test the final model, output the example images and statistical results.
# Auxiliary program
lablel.py: Convert the ground truth picture into a recognition frame, each ground truth picture corresponds to a txt output result, which contains four values: center point coordinates (x, y), recognition frame size (w, h)
generate_train/test: Extract image path
yolov3_customer.cfg: Based on yolov3.cfg in the darknet framework, I modified some of its parameters to improve the performance of the model.
## Results
https://drive.google.com/file/d/1-D8LpGI4TGFUFmvUnLurzplFa5eUUu-k/view?usp=sharing
This link is my final model file, because it is 200M, I can not upload it to github.

![example2](https://user-images.githubusercontent.com/87461237/139568968-d2515a49-4b40-41cc-b843-981636372109.png) ![ISIC_0013766_segmentation](https://user-images.githubusercontent.com/87461237/139569088-52d81d68-eb27-4d45-b1e4-c74a30e2ecb2.png)

The first picture is a prediction of my model on the test set. The melanoma is identified in the pink box. The second picture is the ground truth of the test set. Compared with the first picture, it is found that the two pictures have similar recognition of melanoma. This kind of prediction is relatively successful, and the prediction result of iou is relatively high.

![1635657996(1)](https://user-images.githubusercontent.com/87461237/139569309-c87818cb-d623-4f8d-b2bf-19cd538ed174.png)

This is the statistical results of the prediction. The iou_threshold is set at 0.8, which means that only the predicted recognition box is similar with the ground truch image, the predicted result can be seen as true prediction. Average iou is 63.45%, that is not a high performance. However, 0.72 f1-score is good, which means most of the real melanomas are correctly identified, and there are only a few identification errors in the pictures that are identified as melanomas. However, the 0.25 confidence interval is too large, which decreases the credibility of my result.

## Referrence
[1] AlexeyAB(2018) https://github.com/pjreddie/darknet 

[2] theAIGuysCode(2021) https://github.com/theAIGuysCode/YOLOv3-Cloud-Tutorial 

[3] Ayoosh Kathuria(2018) https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b

[4] Vidushi Meel(2021) https://viso.ai/deep-learning/yolov3-overview/

[5] qqwweee(2018) https://github.com/qqwweee/keras-yolo3

[6] JosephRedmonAliFarhadi(2018) YOLOv3:AnIncrementalImprovement
