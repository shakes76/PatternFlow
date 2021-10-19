# Improved UNet for ISIC2018 image segmentation

## Objective
This project is a practical work for COMP3710 in 2021. The project objective is to implement the improved UNet for ISIC2018 image segmentation. 

## Model Architecture
UNet is the model for biomedical image segmentation. <sup>[1]</sup> Figure1 shows the improved UNet architecture. <sup>[2]</sup> The improved model newly added the context module, 3x3 convolution layer with stride = 2 instead of max pooling layer, localization module, and segmentation layer extracted from localization layer. 

In downsampling part, context module works as residual blocks and the output from the module is concatenate to the input for the localization modules in upsampling block. 3x3 stride 2 convolution works for downsample block. In upsampling, segmentation layer outputted from localization is added to the next segmentation layer.

<p align="center">
  <img width="750" height="350" src = https://user-images.githubusercontent.com/85863413/137878529-a434ecb5-6331-418e-a1cc-892d1ad480c6.png>
</p>

<p align="center">
  <b>Figure1: The improved UNet model architecture</b><sup>[2]</sup>
</p>

  
This repository includes the below files for the improved UNet:
* **IUNet_criterion.py:** This file consists of the two criterion functions: dice coefficient and dice loss. The functions are utilised for training and evaluating the model through the forward and backpropagation steps.
* **IUNet_dataloader.py:** This file is concerning data preparation for UNet model, which works for data loading, data transformation, preparing data loader.
* **IUNet_train_test.py:** This file works to train the model and to assess the segmentation performance with dice coefficient and dice loss. The function in the file returns lists recorded the criteria values by epochs: TRAIN_DICE, TEST_DICE, TRAIN_LOSS, TEST_LOSS.
* **ImprovedUNet.py:** This file includes the classes to build the improved UNet model. The classes with Context, Localization, Up-sampling, Segmentation, and Convolution to down-sampling, and Improved Unet are provided. In this model, Sigmoid function for the binary classification between mask and non-mask areas is utilised instead of softmax function.
* **main.py:** The file performs all procedures for the project, data preparation, training model, and model evaluation. The file includes the parameters with the improved UNet: FEATURE_SIZE, IN_CHANEL, OUT_CHANEL, IMG_TF, MASK_TF, BATCH_SIZE, EPOCHS, and LR. The image size is resized into 128x128 as the initial parameter. Also, random_split() is set to split the dataset into train set and test set by 80:20. The file applies Adam as an optimizer.
* **visualise.py:** The file contains the four functions for plotting the test result and training dice coefficient and dice loss by epochs, and output the segmented images with the predicted mask.

## Dataset
ISIC 2018 Task1 is a dataset with skin cancer images shared by the International Skin Imaging Collaboration (ISIC). <sup>[3]</sup> The dataset consists of 2594 images and 2594 mask images, respectively. The dataset is split into the train set and test set by train ratio = 0.8. 

## Usage model
“main.py” calls all files in the repository to train the model and evaluate the performance. ISIC dataset is needed to be set in the same directory, including main.py.


### Dependency
The model training and evaluation was executed under the environment.
* Pytorch 1.9.0+cu111
* Python 3.7.12
* Matplotlib 3.3.4


## Results
### Dice coefficient and loss
The figure is about train and test dice coefficient and losses by 50 epochs. The test dice coefficient was approximately 0.85 and the test accuracy was stable after 15 epochs.

<p align="center">
  <img width="400" height="350" src = https://user-images.githubusercontent.com/85863413/137879941-5fee3dbe-f873-4dc7-9a21-cd782b50e8c3.png>
</p>

<p align="center">
  <b>Figure2. Dice coefficient</b>
</p>


The test dice loss was stable at roughly 0.15 while train loss declined after epoch 15.

<p align="center">
  <img width="400" height="350" src = https://user-images.githubusercontent.com/85863413/137879979-8a230c8a-d854-47fc-9f34-ad00df4492bf.png>
</p>

<p align="center">
  <b>Figure3. Dice loss</b>
</p>


### Segmentation
The trained UNet predict the mask from the image. The segmentations in the right-hand side column are the images covered with the predicted mask.

<p align="center">
  <img width="500" height="700" src = https://user-images.githubusercontent.com/85863413/137880007-b912725a-be69-4316-8168-54c96be59b75.png>
</p>

<p align="center">
  <b>Figure4. Segmentation</b>
</p>


## References
[1]	Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. https://arxiv.org/abs/1505.04597

[2]	Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, K. H. (2017, September). Brain tumor segmentation and radiomics survival prediction: Contribution to the brats 2017 challenge. In International MICCAI Brainlesion Workshop (pp. 287-297). Springer, Cham. https://arxiv.org/pdf/1802.10508v1.pdf

[3]	ISIC 2018 Task1 https://paperswithcode.com/dataset/isic-2018-task-1




