# Retrain YOLOV3 model on ISIC dataset for lesion detection

## What is YOLO
Yolo is a real time object detection algorithm that uses convolutional neural network (CNN) for doing object detection, and is popular for high accuracy and high speed. The algorithm applies one CNN to the whole image and then segments the image into multiple regions.
It creates bounding boxes and probabilities of existing object in that region.then the bounding boxes are weighted by the probabilities which are predicted. The algorithm only does one forward propagation pass through the CNN to make the predictions. It uses the Non-max suppression mechanism to make sure that each object only detected once. The output of the algorithm is bounding boxes around the detected objects.

## what is this Project about

In this project, I have used a pretrained YOLOv3 model from a public repository in github, and re-trained it on ISIC dataset. Then I have used the re-trained model weights to detect lesion object in ISIC images. To do this project, I have taken the following steps:
1.	creating annotation files from ISIC ground truth images
2.	splitting the data for train and test and creating two text files, train.txt and val.txt, which lists the path of the training and testing data respectively
3.	Creating a folder structure for input data
4.	Re-training Yolov3 pre-trained model on ISIC dataset
5.	using the weight which is trained during previous step to detect lesion in ISIS dataset.

In following each of these steps are explained in details.


## Creating annotation files
I have written annotation.py script to create bounding box and annotation files. This script finds the min and max point x and y values where the pixel value is not 0, draws a bounding box and creates one .txt file for each image. Each annotation file has the same name as its corresponding image file with different extension. Each annotation file contains only one row with the below format:
* Class   x  y  width height
	* The class is the class_id of the object, x and y are coordinates of the center of the bounding box, width is the ratio of the bounding box width to the image width and height is the ratio of the bounding box height to the image height.

## Splitting the data for train and test
I have written train_test_split.py script to split the dataset to train and test. 10 percent of image data is selected and listed in val.txt file, and 90 percent of image data is selected and listed in train.txt file. I have used the 90 percent of data for training, as the more training data we have, the more robust model we train.
 The generated train.txt and val.txt files contain the path of the training data and validation data respectively.


## Creating data Folder structure:
I have created the below folder structure for the input data:
* Zahra_YOLO(Main folder of the project files)
* zahra_YOLO\data\ISIS\images
* zahra_YOLO\data\ISIS\labels
* zahra_YOLO\data\ISIS\train.txt
* zahra_YOLO\data\ISIS\val.txt

All the ground truth images are put in the images folder.
The annotation files are placed into the labels folder.
The files which contain the path of the training and testing data are placed in ISIS folder. (train.txt, val.txt)


## Cloning Yolov3 pretrained model from public repo
I have cloned a public repository from  [github](https://github.com/cfotache/pytorch_custom_yolo_training.git) and I have used these modules from this repo:

* Models.py
	* Creates the YOLOv3 model
* Utils.py
	* Creates the functions to compute the IOU, model performance metrics and create non-max_suppression function to filter detected bounding boxes
* Parse_config.py
	* This script, is parsing the configuration files and extract the config parameters.

To re-train the model and detecting the lesion object in ISIC dataset, I have written the below scripts:
* preprocess.py
	* This script resizes the input image data to 416*416 square size without changing the aspect ratio of the image. In this script the size of the images are adjusted by adding padding to the top, bottom,left and right side of the image.	
	* This script also extracts the  coordinates of the bounding boxes from the annotation file for the unpadded and unscaled images, then updates the coordinates of the bounding boxes according to the scaled and padded image.
* train.py
	* This script uses the YOLOV3 pretrained weights which is downloaded from [web](https://pjreddie.com/media/files/yolov3.weights) and re-trains the YOLOv3 model on the ISIC dataset. The script saves the trained model weights into chechpoint folder after each epoch. After a certain number of epochs, the last saved weight file is used for object detection.
* detection.py
	* This script resizes the input image data and then run the model in inference mode to detect lesion object. the script uses the non maximum suppression mechanism to filter down the number of detected bounding boxes.the NMS mechanism, firstly removes all the bounding boxes that have a detection probability less than the given confidence threshold.	then it excludes all the bounding boxes which have Intersection Over Union (IOU) value higher than the given IOU threshold.
* main_driver.py
	* this script firstly initializes the parameters for re-training the model on ISIC dataset and then call train_model function to perform training.To perform training and then run the detection using the generated weight in the training step, please take following steps:
		* download the pretrained YOLO weight and put it into the directory specified in the Weights_path parameter.
		* comment out the script from line 50 to the end and just run the training.
		* After the training run is finished, then uncomment the commented section in previous step and comment out line 44 which calls train_model func.
		* you also need to replace the pretrained weight file with the weight file generated in previous step. For each epoch a weight file is generated, please pick up the latest one.
		* in line 58, image_path parameter has to be updated by the path of a test image which can be copied from val.tx file.
	* the ouput of the function is a predicted bounding box around the detected lesion.
	* this is a picture of the predicted bounding box around one lesion image.
	![](https://github.com/zaragolshani/PatternFlow/blob/topic-recognition/recognition/zahra_YOLO/150150.png ) 



