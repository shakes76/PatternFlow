# Segmentation of the ISICs dataset using Improved UNet
Published by the International Skin Imaging Collaboration (ISIC), the ISICs 2018 dataset
was used as part of the Skin Lesion Analysis Towards Melanoma Detection recurring challenge.
This challenge is an opportunity for participants to develop automated tools which aid in the
diagnosis of melanoma. This project employs the Improved UNet [1] architecture to perform
segmentation on the aforementioned ISICs dataset. 

## ISICs dataset

### Data preprocessing

### Segmentation

## Architecture

### Improved UNet
Proposed in 2018 [1], the Improved UNet is designed upon the original model of UNet, proposed in 2015 [2]. 

_Image of architecture_
_Figure 1: Improved UNet Architecture [1]_

The Improved UNet is composed of two main sections, the encoding path and the decoding path.

## Optimizer & Loss

### Dice Similarity Coefficient

## Results

## Dependencies

## References
[1]: Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., Maier-Hein, K.H, "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge". _arXiv: Computer Vision and Pattern Recognition_, 2018.

[2]: Ronneberger, O., Fischer, P., Brox, T., "U-net: Convolutional networks for biomedical image segmentation,". _International Conference on Medical Image Computing and Computer-Assisted Intervention_, 2015. (Springer, pp. 234-241).
