## The Report
### Name : **Sanchit Jain**
### Student Number : **s4746168**

Hi everyone, 
This is the final demo (i.e., report assessment) for my course.
I am doing the **TASK 1** *to segment the ISIC data set with the improved UNet.*
*Also, all the labels must have a minimum Dice similarity coefficient of 0.8*.

U-Net architecture was introduced by Olaf Ronneberger, Philipp Fischer, 
Thomas Brox in 2015 for tumor detection but since has been found to be 
useful across multiple industries This U-shaped architecture mainly 
consists of Convolutional layers implemented in a specific way. No Dense 
layer is presented in model. 

U-Net Image Segmentation in keras by Margaret Maynard-Reid 
[https://929687.smushcdn.com/2633864/wp-content/uploads/2022/02/1_unet_architecture_paper-768x427.png?lossy=1&strip=1&webp=1]

ISIC 2017 dataset will be used to train and will be implemented this 
model and identify the skin tumour. To access and download the ISIC 2017 
dataset visit : [https://challenge.isic-archive.com/data/#2017]. 

For this model you need to download all the data files i.e., Normal data 
files and all Ground Truth data as well. After downloading the data 
files unzip them and then path of these data files needs to be added in 
the dataset.py file. **Make sure you specify and locate the correct** 
**files to correct data variables or else the model might not work** 
**properly**. I have added location for my data files and hence after 
downloading ***dataset.py*** it might show error on your computer unless 
you specify the correct path of the data files.

The ***modules.py*** present in the directory consist of full architecture of 
UNET. Each component in this file is implemented as a function. For the 
UNET architecture, I have created **4 separate functions**, one which consists 
of _all convolutional layers_. _Down Sampling block_ which each time 
implements the **down sampling of the data**, an _up sampling block_ 
reponsible for **upsampling of the data** and the last _build unet model_ 
function which **calls each of the other functions** and builds the whole 
architecture of the UNET model and **returns the model**.
