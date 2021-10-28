# Segmentation of the ISICs dataset using Improved UNet
Published by the International Skin Imaging Collaboration (ISIC), the ISICs 2018 dataset
was used as part of the Skin Lesion Analysis Towards Melanoma Detection recurring challenge.
This challenge is an opportunity for participants to develop automated tools which aid in the
diagnosis of melanoma. This project employs the Improved UNet [1] architecture to perform
segmentation on the aforementioned ISICs dataset. 

## ISICs dataset
The ISIC's dataset contains 2594 images and their associated segmentation masks. By default, the dataset contains images of varying sizes and aspect ratios. Data normalisation and resizing was applied to provide consistency throughout the inputs.

_Image of Training data / Image of associated mask_
_Figure 1: Sample image and associated mask from the ISICs 2018 dataset_

### Data preprocessing
As part of the pre-processing phase, all of the images (training images and masks) were normalised. In order to be run through the network, all of the images had to be the same size. The size chosen was (192, 256). The training images kept 3 colour channels: [192, 256, 3]. On the other hand, the segmentation masks were reduced to a single colour channel: [192, 256, 1]. The segmentation masks were also thresholded: pixels with a value > 0.5 after normalisation were set to 1, and the rest were set to 0. 

#### Training, Test & Validation Split.
The Training, Testing and Validation data split that was chosen was 70 / 15 / 15. Some research was conducted on the optimal split for medical data. In general, it was found that there is no single correct split, however this percentage seemed to be the most widely used. For a dataset of this size, that means there was around 1800 training samples, and 390 training & validation samples.  

## Architecture
Proposed in 2018 [1], the Improved UNet is designed upon the original model of UNet, proposed in 2015 [2]. 

_Image of architecture_
_Figure 2: Improved UNet Architecture [1]_

The Improved UNet is composed of two main sections, the encoding path and the decoding path.

### Context Modules

### Localisation Modules

### Skip Connections

## Optimizer & Loss
The optimizer used in this implementation was the Adam optimizer with a learning rate of 5e-4.

### Dice Similarity Coefficient
The Dice Similarity Coefficient is a common metric used in segmentation problems.

## Results

### Model Output

### Accuracy & Loss

## Usage
To run this network, ensure you have the appropriate Dependencies installed. 

Download the ISIC's 2018 dataset and place the training images and segmentation masks in two separate folders in the directory where the `model.py` and `driver.py` are located, named as so:
- Training images: ISIC2018_Task1-2_Training_Input_x2 
- Segmentation masks: ISIC2018_Task1_Training_GroundTruth_x2

Open up a commandline and navigate to the directory where `driver.py` is saved, and run it:

`python driver.py`

To ensure the data was loaded correctly, an image from the Training Input should appear on-screen, followed by its corresponding mask from the Training GroundTruth. 

## Dependencies
- Python _
- Tensorflow _
- Matplotlib _

## References
[1]: Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., Maier-Hein, K.H, "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge". _arXiv: Computer Vision and Pattern Recognition_, 2018.

[2]: Ronneberger, O., Fischer, P., Brox, T., "U-net: Convolutional networks for biomedical image segmentation,". _International Conference on Medical Image Computing and Computer-Assisted Intervention_, 2015. (Springer, pp. 234-241).
