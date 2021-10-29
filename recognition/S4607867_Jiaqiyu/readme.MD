# Yolo V3 Object Detection （based on AlexyAB darknet）

# Task introduction：
The task I choose for COMP3710 report is Yolo Object Detection task.
It is widely used algorithm in computer vision tasks such as image annotation,vehicle counting, activity recognition,or face detection.
The task here is to identify the Lesion Objects on patients' skin pictures. 


# Step1 building environment：
I have cloned Datknet library from https://github.com/AlexeyAB/darknet .
In order to build the yoloV3 model, 

I am using Windows10 as OS, so I followed AlexeyAB's tutorial - compile on Windows part.


# Step2 preparing support files： 
I have downloaded ISIC challenge 2017 as the target dataset.
It contains 2000 train samples, 150 validation samples, and 600 test samples.
The ground truth of lesion objects are stored as segmentation shadows. 
So I wrote a python script to transform those images into the required format of input labels.

The darknet also requires us to set some parameters for the network.
Here is the configuration of my yoloV3 network:


# Step3 training yoloV3 model：





The trained weights is stored in [ url  ]


# Step4 validation on test set:

darknet has a build-in validation command, which can returns the average IOU ratio and other evaluation metrics.


# Conclusion


















# PatternFlow
Pattern recognition and image processing library for Tensorflow (TF)

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* algorithms
* recognition problems

In the algorithms, you will find many image processing algorithms ported to TF
* denoise algorithms 
* image processing algorithms 
* transform algorithms 
* metrics
* histograms
etc.

In the recognition folder, you will find many recognition problems solved including:
* OASIS brain segmentation
* Classification
etc.

