# Improved UNet for ISIC2018 image segmentation
The project is the practical work for COMP3710 in 2021. This report will summarise the information with the improved UNet model in the repository.


## Objective
The project objective is to implement the improved UNet for ISIC2018 image segmentation. UNet is a developed model for biomedical image segmentation, which automatically identifies the tumour area.<sup>[1]</sup> The automatic image segmentation without objective will support the medical and experimental works, while the higher accurate image segmentation performance is also required. This project aimed to implement the improved UNet model for Brain tumours into the ISIC2018 image dataset.


## Model Architecture
Figure 1 shows the improved UNet architecture for Brain tumours.<sup>[2]</sup> The improved model utilises the context module, 3x3 convolution layer with stride = 2 instead of max-pooling layer, localisation module, and segmentation layer extracted from localisation layer. In down-sampling part, context block works as residual blocks of ResNet. The context block consists of 3x3 convolution layer, batch normalisation layer, dropout layer, and activity function layer. LeakyReLu is applied as the activation function in the model. The output through the context block is concatenated to the input for the localisation modules in up-sampling block. After that, the output features are decreased with 3x3 convolution layer for the following context block.

In up-sampling, the concatenated input is fed into the localization block. Then, the output from the localization is fed into the convolutional layer to transform into a segmentation layer to add to the next segmentation layer and the up-sampling block, respectively.

<p align="center">
  <img width="750" height="350" src = https://user-images.githubusercontent.com/85863413/137878529-a434ecb5-6331-418e-a1cc-892d1ad480c6.png>
</p>

<p align="center">
  <b>Figure1: The improved UNet model architecture</b><sup>[2]</sup>
</p>

In terms of the loss function, dice loss is utilised for UNet. The loss function is represented as 

<p align="center">
  <img width="180" height="30" src = https://user-images.githubusercontent.com/85863413/138026696-fd7fd35e-bc0b-4b16-8b63-bc67f5011c78.png>
</p>

Dice coefficient is represented as 

<p align="center">
  <img width="120" height="30" src = https://user-images.githubusercontent.com/85863413/138026430-b43c30cf-100d-4a29-b8c7-094adb299d17.png>
</p>

Dice coefficient measures the similarity between the target mask and the predicted mask from the model.


## Files
This repository includes the below files for the improved UNet.

**criterion.py:** This file consists of the two criterion functions: dice coefficient and dice loss. The functions are utilised for training and evaluating the model through the forward and backpropagation steps.

**dataloader.py:** This file is concerning data preparation for UNet model, which works for data loading, image augmentation, transformation into data loader.

**model_train_val.py:** This file works to train the model and to assess the segmentation performance with dice coefficient and dice loss. The function in the file returns lists recorded the criteria values by epochs: TRAIN_DICE, VAL_DICE, TRAIN_LOSS, VAL_LOSS.

**model.py:** This file includes the classes to build the improved UNet model. The classes with Context, Localization, Up-sampling, Segmentation, and Convolution to down-sampling, and Improved UNet are provided. In this model, Sigmoid function for the binary classification between mask and non-mask areas is utilised instead of softmax function.

**driver.py:** The file performs all procedures for the project, data preparation, training model, and model evaluation. The file includes the parameters with the improved UNet: FEATURE_SIZE, IN_CHANEL, OUT_CHANEL, IMG_TF, MASK_TF, BATCH_SIZE, EPOCHS, and LR. The image size is resized into 128x128 as the initial parameter. Also, random_split() is set to split the dataset into train set, validation set, and test set by 50:25:25. Adam is applied as an optimizer to train the model.

**visualise.py:** The file contains the five functions for plotting the test result and training dice coefficient and dice loss by epochs, and output the segmented images with the predicted mask.


## Dataset
ISIC 2018 Task1 is a dataset with skin cancer images shared by the International Skin Imaging Collaboration (ISIC). <sup>[3]</sup> The dataset consists of 2594 images and mask images, respectively. The dataset is split into the train set, validation, and test set by ratio:  0.5: 0.25: 0.25. 


## How to run
“driver.py” calls all files in the repository to train the model and to evaluate the performance. ISIC dataset is needed to be set in the same directory including the files. After that, put the command ‘python driver.py’ in the terminal and execute the command.


## Dependency
The model training and evaluation was executed under the environment.
* Pytorch 1.9.0+cu111
* Python 3.7.12
* Matplotlib 3.3.4



## Results
#### Dice coefficient and loss
The figure is about train and validation dice coefficient and losses by 50 epochs. The validation dice coefficient was approximately 0.85 and it was stable after 15 epochs.


<p align="center">
  <img width="400" height="300" src = https://user-images.githubusercontent.com/85863413/138023981-96eeecf9-3bbb-4e6e-a4f9-e7376b7dd216.png>
</p>

<p align="center">
<b>Figure2. Dice coefficient</b>
</p>


The validation dice loss was stable at roughly 0.15 while train loss declined after epoch 15.


<p align="center">
  <img width="400" height="300" src = https://user-images.githubusercontent.com/85863413/138024051-ecd6ef76-3c51-4493-abdd-8c04d6dd8d26.png>
</p>


<p align="center">
<b>Figure3. Dice loss</b>
</p>



#### Segmentation
The trained UNet predict the mask from the image in test set. The segmentations in the right-hand side column are the images covered with the predicted mask. The dice coefficient of the image is provided in the label. The dice coefficients in the figure recorded over 0.87.
 

<p align="center">
  <img width="500" height="700" src = https://user-images.githubusercontent.com/85863413/138024077-2d17b2fd-fb1c-4ff9-8030-8ea7521f2420.png>
</p>


<p align="center">
<b>Figure4. Segmentation</b>
</p>



## References
[1]	Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. https://arxiv.org/abs/1505.04597

[2]	Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, K. H. (2017, September). Brain tumor segmentation and radiomics survival prediction: Contribution to the brats 2017 challenge. In International MICCAI Brainlesion Workshop (pp. 287-297). Springer, Cham. https://arxiv.org/pdf/1802.10508v1.pdf

[3]	ISIC 2018 Task1 https://paperswithcode.com/dataset/isic-2018-task-1
