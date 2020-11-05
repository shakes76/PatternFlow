# Image Segmentation Task
Task: Segment the OASIS brain data set with an Improved UNet with all labels having a minimum Dice
similarity coefficient of 0.9 on the test set.

Data resource: magnetic resonance (MR) image of the brain from OASIS dataset.

Dataset contains train data and  seg_train data

x_train = train data
y_train =  seg_train data

![](images/example.png)

# Load Dataset
For x_train data, we have 9664 images which are 256*256 pixels(black and white photo)


We will make 4 labels which are:
* 0 - Background
* 1 - CSF (cerebrospinal fluid)
* 2 - Gray matter
* 3 - White matter

# For example:
![](images/labels.png)
