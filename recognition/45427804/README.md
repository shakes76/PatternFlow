# Image Segmentation Task
Task: Segment the OASIS brain data set with an Improved UNet with all labels having a minimum Dice
similarity coefficient of 0.9 on the test set.

Data resource: magnetic resonance (MR) image of the brain from OASIS dataset.

# Load Dataset
x_train dataset has 9664 images which are 256*256 pixels(black and white photo)

x_test dataset has 544 images which are 256*256 pixels(black and white photo)

y_train (labels) has 9664 images which are 256*256 pixels(black and white photo)

y_test (labels) has 544 images which are 256*256 pixels(black and white photo)

![](images/example.png)

# Data preparation

 Using **sorted function** to sort each dataset to make sure that each picture can correspond to the corresponding label. After that, using **reshape function** to make training data and testing data into 4 dimensions for instance (9664,256,256,1) and (544,256,256,1), and then divide training data and test data by 255 to achieve **normalization**.
 


We will make 4 labels which are:
* 0 - Background
* 1 - CSF (cerebrospinal fluid)
* 2 - Gray matter
* 3 - White matter

# For example:
![](images/labels.png)
