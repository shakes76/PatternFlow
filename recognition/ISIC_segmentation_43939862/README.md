# ISIC Image Segmentation Using Improved Unet

Improved Unet[1] is a ConvNet which is adapted from the original Unet[2], with a few added features to improve performance. The two main additions in Improved Unet is the use of residual connections in the encoder, and the use of segmentation layers in the decoder. The residual connections help with gradient-vanishing by having activations skip layers and be added element-wise to the activations of the skipped layer[3]. Segmentation layers [4] allow information from throughout the decoder to be used in the output layer directly, reducing loss of information from upsampling[4]. In this project, Improved Unet is applied to the ISIC 2018 (task 2) dataset, with the goal of segmenting the skin cancer images with an average DICE score of at least 80% on the test set.

## The Algorithm


## Dependencies


## Training/validation/test split


## Output




[1] Improved Unet ref
[2] Unet ref
[3] Residual blocks ref
[4] Segmentation layers ref