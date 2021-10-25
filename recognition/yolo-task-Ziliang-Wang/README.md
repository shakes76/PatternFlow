# Yolov3 ISIC dataset lesion detection

The skin is the largest organ of the human body. Some skin diseases, such as malignant black, melanoma. These diseases are likely to cause death, and the consequences of misdiagnosis will be very serious. Moles are skin diseases with mild symptoms, but some fatal skin diseases such as malignant black and mildly symptomatic skin diseases and their Similar. Effectively locating and detecting these skin diseases plays a critical role in saving the lives of patients. The ISIC data set contains 23k melanoma examples of malignant and benign images that classify skin injuries. Yolo is a kind of slider image The detection technology is different from mask-rcnn in that it uses a one-stage algorithm, which only uses a convolutional neural network to directly predict the classes and positions of different targets. In here, YOLOV3 is used to detect the lesion area of the ISIC data set. The goal of this task is that all detections having a minimum Intersection Over Union of 0.8 on the test set.


### ISIC dataset
In this task, we have given the preprocessed data set, which includes 2594 pictures and the corresponding segmented black and white pictures. 
![ISIC_0016037](https://user-images.githubusercontent.com/65603393/138678240-f794ef8b-b534-4a91-a953-96f2cb366411.jpg)
*Figure 1: Training image.*

![ISIC_0016037_segmentation](https://user-images.githubusercontent.com/65603393/138678838-ce003a09-fb42-41b2-96dd-31819110ff42.png)
*Figure 1: Segmented image.*
