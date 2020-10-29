
# Problem:
This implementation solves the recognition problem to segment the ISIC dataset with an improved Unet architecture 

## Dataset description:
The dataset contains both segmented and raw images of skin patches to detect areas with melanoma(skin cancer). The aim of this deep neural network is to train on a subset of this dataset and then segment test images to identify areas with melanoma.The images in the dataset were of different sizes, which was then modified in the pre-processing stage to fit into the model.

## Inputs and Preprocessing:
1. To read the files of the images, glob library of python has been used here. After that, the files were sorted in an ascending index order so that there is no mismatch and input and target labels.
2. All the images have been resized to (256,256) size to have uniformity.
3. The segmented images were stored as y or target variable and others as x.
4. As the target variable,y, had various unique labels from 0 to 255 owing to different representations in grey scale, the variable was scaled and rounded off so that the unique values were 0(background or normal skin) and 1 (patch or detected melenoma area).
5.For splitting the data into train, validation and test data, sklearn's 'train_test_split' function has been used, herein, 20% data was left out as test data and 80% was separated as training data( out of which 25% was used as validation data).A random state was also used for reproducibility of results.


## ModelArchitecture
The model used for this classification is an improved u-net structure taken from the research paper as cited in the *reference* section at the end.
The architecture image is as follows-
![Screenshot](architecture.PNG)

## Model Parameters


## Model result evaluation 

## Implementation timeline
## REFERENCE
F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and
Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available:
https://arxiv.org/abs/1802.10508v1

