# Improved UNet for ISICs dataset - Task 1

#####################################################################################################

This project is created for COMP3710 Pattern Recognition Report. 

This project aim to segment the ISICs data set with the Improved UNet model with all labels having a minimum Dice Similarity Coefficient (DSC) of 0.8 on the test set - Task 1 

This project is finished by Xinwei Li, 44935209.

#####################################################################################################

### Introduction

### Methodology

### Project Structure

In this repository, there are serveral Python scripts used for different conditions:

- train.py - Trainning script. 
- test.py - The main driver script when running the model. 
- Data_Loader.py - Image data preprocessing. Getting individual transformations and data (as a Dict)
- image.py - Imgae data preprocessing, creating 2D image from 3D.
- Metrics.py - Calculates the dice coefficient for the images and getting the accuracy of the model.
- losses.py - Calculating the dice loss and metrics
- Models.py - The implemenation of the improved U-Net model
- ploting.py - Generating the output images and metric plot.
- Unet_ISICs_JupyterBook.ipynb - Additional Jupyter Notebook for displaying the esssential process step by step.

### Dependencies

1. python>=3.6
2. torch>=0.4.0
3. torchvision
4. torchsummary
5. tensorboardx
6. natsort
7. numpy
8. pillow
9. scipy
10. scikit-image
11. sklearn

### Dataset
Assuming users who run this model have downloaded and unzipped the ISICs 2018 dataset.

### Training

### Testing

### Result

### Reference
