# Skin Mole Segmentation on the ISIC2017 data set using the improved UNet.

## Author
Name: Michael Smith
Student Number: 45838464

## Problem Description
Image segmentation is important in the current scene of medicine as it allows patients to gain important information without the immediate presence of a doctor. This is especially beneficial for people who don't have reliable access to a hospital or GP. Image segmentation is the process of separating an image into its constituent classes. In this example the two classes are the skin and the mole. The goal is to segment a test data set of lesions and compare it to the manually segemented images. The degree of similariy is the dice coeffiecient (described here https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and this model should aim to achieve a dice coefficient higher than 0.8.  

## Why Improved UNet and How it Works
The improved UNet model is an autoencoder network that provides boosts in performance over the original UNeting: "... architecture of the context pathway, normalization schemes, number of featuremaps throughout the network, nonlinearity and the structure of the upsampling
pathway". Both networks are made up of encoder and decoder submodules, the input image is encoded down to a low dimension latent space, then is decoded up to segmentation map. The detailed architecture is shown in the figure below. 

![image](./report_images/improved-unet-diagram.png)

Each block can be built by layers provided in the tensorflow.keras.layers library. The details for the blocks (context, localisation, segmentaion etc.) are in modules.py. These submodules are then combine into the full ImprovedUNet model which is at the end of modules.py.

The  An important difference between this model and the paper model is that this problem is using 2D data not 3D. The input is a 128x128x3 array (3 for r,g,b channels) and the ouput is 128x128x2 (2 for each class).  

## Data Preprocessing
The ISIC data needs to be preprocessed in order for optimal performace of the model. Firstly the "superpixel" files were removed using script commands. The image files were transformed to tensorflow arrays using the io and image libraries from tensorflow. The data preprossing function are located in dataset.py. Fortunately the ISIC 2017 data has already been seperated into training, validationa and testing splits and so they can be used directly. Training size is 2000, validation size is 150, testing size is 600. Note, once the data has been preprocessed once it helps to save these arrays using np.save(). This will prevent wasting time for preprocessing. These arrays can be loaded with np.load().

## Training Results
The model was trained with a batch size of 32 and run for 30 epochs. The dice similarity is shown to be sufficiently heigh. 

![image](./report_images/dice_accuracy.png)

![image](./report_images/loss.png)

## Dependecies
python 3.9.13
tensorflow 2.9.1
matlplotlib 3.5.2 
numpy 1.23.3 

To use GPU when training

tensorflow-gpu 2.6.0

## References
Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., &amp; Maier-Hein, K. H. (2018, February 28). Brain tumor segmentation and radiomics survival prediction: Contribution to the brats 2017 challenge. arXiv.org. Retrieved October 19, 2022, from https://arxiv.org/abs/1802.10508v1 