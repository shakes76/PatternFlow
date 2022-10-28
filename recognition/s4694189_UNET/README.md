<h2> Name: Ojas Madhusudan Chaudhari </h2>

<h2> Student Number: 46941893 </h2>

<h2>Student Email: o.chaudhari@uqconnect.edu.au </h2>

<h2>Project: Segmentation using UNet </h2>


<h1> Segmentation of ISIC data with improved UNet </h1>

This project develops a solution to the ISIC challenge using UNet. The objective of the project is to train convolutional neural network to segment the ISIC images using improved UNet. The architecture of improved UNet has been referred from F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein’s “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” research paper. This paper gives the architecture of improved UNet. It consists of two parts: -
* Downsampling
* Upsampling

Following diagram exactly depicts the architecture
![Improved UNet](https://github.com/OjasChaudhari23/PatternFlow/blob/topic-recognition/recognition/s4694189_UNET/improvedunet.png)

The neural network starts with giving an input layer of 256*256*3. The actual size of the images are 128*128. The images are then resized, normalized and given as array to the network. The neural network starts with contraction at start. It starts with 16 layers and increases in each step. Each step consists of two equal features. After it reaches to 256, Expansion starts. In this project Transposecv has been used for the expansion.

<h2>Running the program: </h2>
Please download the modules.ipynb file to run the program or download al =l the files as they are related with each other

<h2> File structure </h2>
<b> dataset.py </b>:- This file loads the dataset and give the x and y array values with the pre-processing of images.
<b> module.py </b>:- This file has UNET neural network defined in it and it returns the model
<b> train.py </b> :- This file trains the model and plots the result. The model uses dice similarity measure to evaluate the model's performance.
<b> predict.py </b> :- This file evaluates the performance on test data.

<h2> Plots and visualizations
 After building the model. coefficient loss of training and validation data has been visualized with respect to epochs
 ![Loss](https://github.com/OjasChaudhari23/PatternFlow/blob/topic-recognition/recognition/s4694189_UNET/loss_picture.jpg)
 
 Similarly dice coefficient values have been visualized
 ![Dice](https://github.com/OjasChaudhari23/PatternFlow/blob/topic-recognition/recognition/s4694189_UNET/dice_coefficient.jpg)
 
 In this project 3 data files which have been given in 2017 ISIC Challenge have been used. Train, validation and test images with their masks have been used for the study. Also the result on test data is quite good and it is more than 0.8
 ![evaluation](https://github.com/OjasChaudhari23/PatternFlow/blob/topic-recognition/recognition/s4694189_UNET/prediction.jpg)
 
 







 

