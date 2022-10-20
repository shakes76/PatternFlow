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

### Information about dataset.py & preprocessing data
<sub>For this model you need to download all the data files i.e., Normal data 
files and all Ground Truth data as well. After downloading the data 
files unzip them and then path of these data files needs to be added in 
the dataset.py file. **Make sure you specify and locate the correct** 
**files to correct data variables or else the model might not work** 
**properly**. I have added location for my data files and hence after 
downloading ***dataset.py*** it might show error on your computer unless 
you specify the correct path of the data files.</sub>

### Information about modules.py
<sub>The ***modules.py*** present in the directory consist of full architecture of 
UNET. Each component in this file is implemented as a function. For the 
UNET architecture, I have created **4 separate functions**, one which consists 
of _all convolutional layers_. _Down Sampling block_ which each time 
implements the **down sampling of the data**, an _up sampling block_ 
reponsible for **upsampling of the data** and the last _build unet model_ 
function which **calls each of the other functions** and builds the whole 
architecture of the UNET model and **returns the model**.</sub>

### Information about training.py
<sub>The **training.py** is file where all the model related process are 
implemented. The file imports a bunch of other python files created such 
as *_dataset.py & modules.py_* for compling and training the model with 
our *_pre processed images & UNet architecture as well_*. Model is being 
called here in the starting. I have **used** two other functions for 
**calculating dice coefficient** that is essential for all the loss and 
metrics. You can refer to : 
[https://github.com/keras-team/keras/issues/3611] for the references of 
the code with **dice coefficient**. While compiling the model I have 
used Adam optimizer. I have fitted the model and then saved it to use 
later. This file contains the source code for training, validating, 
testing and saving your model. All the graphs are also plotted for the 
training of model. These graphs are really important in understanding 
the statistics and journey of the model.</sub>

### About Dice Coefficient
<sub>*Dice coefficient* are _important_ in evaluating the **semantic segmentation** 
**models**. The Dice coefficient is very similar to the IoU. They are 
positively correlated, meaning if one says model A is better than model 
B at segmenting an image, then the other will say the same. *_In dice_* 
*_coefficient we count the similar pixels_* (taking intersection, present in 
both the images) in the both images we are comparing and multiple it by 
2. And divide it by the total pixels in both the images.</sub>

### Information about predict.py
<sub>The **predict.py** contains **trained model** and all the _usage_ of the 
trained model. It is the file where the *_evaluation of model is done_* and 
prints out the results. Through this file one can correlate and *predict* 
the results that model brings.</sub>

### About Plotted graphs
<sub> I have plotted two graphs. Both of these graphs are **plotted** 
**after** the training of model is completed. One graph is for the 
**dice coefficient similaritiy** : 
[https://github.com/Sanchitjain16/PatternFlow/blob/topic-recognition/recognition/s4746168_report_assessment/Plotted_Graph/Dice%20Coefficient.png] 
and the other one is for the **loss** : 
[https://github.com/Sanchitjain16/PatternFlow/blob/topic-recognition/recognition/s4746168_report_assessment/Plotted_Graph/Loss.png]. 

**IMPORTANT NOTE : ** 
**_These graphs are plotted for 3 epochs only and the dataset for these_** 
**_graphs is also very limited and hence they might have higher values in_** 
**_plotting rather than the original graphs when full dataset is loaded._**
