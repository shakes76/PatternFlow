# ISICs Dataset Segmentation with Improved UNet

## Author
**Name:** Jeng-Chung, Lien
**Student ID:** 46232050
**Email:** jengchung.lien@uqconnect.edu.au

## Introduction
The International Skin Imaging Collaboration (ISIC) is an international effort to improve melanoma diagnosis. Melanoma is the deadliest form of skin cancer which is a major public health problem. The objective here is to used an Improved version of the UNet based model inspired from the paper [[1]]() to perform image segmentaion on the ISICs dataset and achieve a [Dice similarity coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) greater than 0.8 on the test set.

## ISICs Dataset
The dataset here to perform the segmentation task is from the [ISIC 2018 challenge](https://challenge2018.isic-archive.com/). The preprocessed version of this dataset contains 2,594 samples. Each sample is a pair of the skin lesion(skin cancer) image and the true label of the skin lesion segmentaion(the target). An example of the sample "ISIC_0000001" in the dataset is shown below, with the skin lesion as the input on the right and the skin lesion segmentation as the target output on the right.

**ISIC_0000001 sample in ISICs dataset example:**
![ISIC_0000001_example.png](ExampleImage/Dataset/ISIC_0000001_example.png)

### Data preprocessing
1. **Image Shape Inconsistent:** The images in the 2,594 samples given above has inconsistent image shapes across all samples. For example images with a heightxwidth of 384x511, 768x1024, etc are observed. To keep information consistent across all samples and to fit the same image shape into the model, a reshape transform of the images are needed. The maximum shape of 256x256 to reshape all images as a square image is calculated as finding the minimum image shape across all samples
2. **Train, Validation and Test Set Split:** A train, validation and test set split is performed on the dataset. A train set 60% of the whole dataset, the train set is the dataset to train the model. A validation set 20% of the whole dataset, the validation set is the dataset to validate how a model performs when training. A test set 20% of the whole dataset, the test set is the dataset to evaluate what is close to the ture performance of the model on unkown data distribution.