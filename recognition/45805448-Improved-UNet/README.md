# Image Segmentation of the ISIC dataset using Improved UNet

by Jaleel Abdur-Raheem, 458054448

## Introduction

### ISIC dataset

The ISIC challenge is a yearly event with the goal of developing imaging tools for diagnosis of skin cancer (i.e. melanoma). The challenge provides a dataset of coloured images of skin lesions, alongside a two-coloured mask (i.e. ground truth) dataset that labels where the lesion is in the image. The aim of this task is to create an algorithm that takes the image dataset and transforms it as close as it can to the mask dataset. This is known as image segmentation.

The [dataset used](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip) and [relevant ground truths](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip) come from the 2016 challenge, which had a much more reasonable downloading duration and size given the time constraints of the COMP3710 report.

### Image Segmentation using Improved UNet

In 2017, [1] created the Improved UNet, which borrows from the original UNet architecture and applies the following changes:

- customised data augmentation (standardisation and cropping)
- using twice the number of filters in the localization pathway (upsampling half of the network)
- applied Leaky ReLU activation to intermediate layers
- adapted the dice loss function for multiple classes

The full structure of the Improved UNet can be seen below:

![Improved UNet Structure](images/improved_unet_structure.png)

In the first half of the network known as the "context aggregation pathway", the network encodes the information using convolutional and adding layers to extract features. Every time this is done, the network enters a new "level" where the intermediate context is saved for later usage in the other half of the network.

In the second half of the network known as the "localization pathway", the network upsamples back to the original image size, while localizing the samples by incorporating the features extracted at every intermediate level in the first half. Additionally, this Improved UNet utilises deep supervision by integrating the segmentation layers alongside some of the upsampling modules, introducing nonlinearity to the network. The upsampling and segmentation paths finally combine at the end to produce the final segmentation results.

## Method

### Preprocessing

In my implementation of the Improved UNet, I apply several augmentations to both the raw images and masks:

- Resizing: the original dataset used images of size (767, 1022). I downscaled this to (192, 256) which is a close-enough "factor" of the original size, and guarantees the down/upsampling done in the network recovers the proper size of the image. This also reduces the training time for the model at relatively little cost of accuracy.
- Normalization: both sets are divided by 255, but in the case of the masks the resulting tensor is cast/rounded to integers then one-hot encoded into two classes (i.e. lesion, non-lesion).
- Random flipping: both sets have random horizontal and vertical flips applied. This generalises the orientation of the lesions.
- Random cropping: both sets are padded by 8 pixels on each side then randomly cropped back to their original size. This generalises the location of the lesions.

The randomisation is controlled between the two datasets by concatenating them along the channels axis. This ensures that the random augmentations on the images are duplicated on the corresponding masks. Afterwards they were returned to their original shapes.

### Split Allocation

The dataset was batched into groups of 5, while the original paper tried groups of 2. I also split the full dataset into training, validation, and testing datasets with a split of 7-1-1. I did this because the full dataset has 900 images, which meant my split was cleanly divided. The split roughly translates to 77.7%-11.1%-11.1%. I found this split to yield desirable results after running on the full dataset.

## Results

## Running the Project

### Steps

Running the project has been simplified to one function in train.py and predict.py each.

To train the model, write a script that calls train.train_isic_dataset(), with parameters:

- images_path: path to the ISIC dataset images
- masks_path: path to the ISIC dataset ground truths
- dataset_path: location of where the preprocessed dataset will be serialised (quicker to load it than re-augment the original datasets)
- model_path: location of where the weights of the trained model will be serialised (useful for doing predictions outside the training environment)
- plots_path: location of where the plots from the project should be stored. This included showing preprocessed samples, model metric performance, and prediction samples.

To predict masks from the model, you can either call predict.predict_isic_dataset(), or (in the same environment as training) pass the trainer returned from train.train_isic_dataset to predict.predict_isic_dataset(). Either use case will ensure the testing dataset and model weights are loaded, then it will predict one batch of testing samples.

All textual outputs of the project are printed to stdout and all figures are saved in the plots_path directory.

### Dependencies

Datasets used:

- ISIC images: https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip
- ISIC ground truths: https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip

Libraries used:

- os
- tensorflow 2.10.0
- keras 2.10.0
- matplotlib 3.6.1

Trained on the Rangpur cluster managed by the University of Queensland

## References

[[1]](https://arxiv.org/pdf/1802.10508) F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, K. H. Maier-Hein, "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge", (2017)
