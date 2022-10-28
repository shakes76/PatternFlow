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

#Running the program:





 

