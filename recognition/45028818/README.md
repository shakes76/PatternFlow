# ISIC Dataset with Improved UNet
***
## Introduction
This aim of this model was to perform image segmentation on the ISIC dataset, this dataset contains images of skin cancer and their respective ground truth mask values. 

## Model Design
The model design is an implementation of the improved U-Net from the paper written by Fabian Isensee et al [1].
The architecture of the model is shown in this image from the paper.
![image](https://user-images.githubusercontent.com/14146158/139621908-c9d467e4-2e76-4a61-a60e-e914604e5c73.png)

Like the original U-Net there is a context aggregation pathway (context pathway) responsible for encoding increasingly abstract representations of the original input as it 'descends' down the layers. Following this pathway is the localization pathway that recombines the aggregation output of that layer with the up-sampled features from the layer below. 

### Context Pathway
The context pathway begins with a 3x3x3 convolution, following this is the first of the context module. The context module consists of seven layers: two 3x3x3 convolutional layers with each convolutional layer followed by instance normalization layer and a leaky ReLu activation layer with an alpha of 0.01. In between each stack of three layers (convolutional/normalization/activation) is a dropout layer with a dropout probability of 0.3.
The final part of the context module is that the seven layers described previously are surrounded by a pre-activation residual block which sums the input into the context module with the output.

The final component of the context pathway is each context module being connected by a 3x3x3 convolutional block with a stride of 2 to reduce the resolution of the feature maps.

### Localization Pathway 
The localization pathway begins after the context pathway has reached layer 5, it begins with an upsampling module which consists of upsampling layer followed by 3x3x3 convolutional block. The output of this upsampling layer is then recombined with the corresponding output of the context pathway on that level, the recombination is via concatenation. Each following level of the localization pathway consists of a localization module followed by an upsampling module and concatenation. With the exception of level 1, which consists of a 3x3x3 convolutional block followed by a segmentation layer.

The localization module consists of a 3x3x3 convolutional block followed by a 1x1x1 convolutional block. 

The output consists of the sum of the segmentation layers from level 1, 2, and 3. The output of this sum is put into a final sigmoid activation layer, this is a different implementation to the paper as it used a softmax layer. 

## Training Details
My model is trained using images resized to 512x512x3, with data augmentation done through Keras's ImageDataGenerator. This augmentation consists of shearing, zooming, and horizontal and vertical flips. The data preprocessing only splits the data into a training set and a testing set, while I would usually do a validation set as well to help guide parameter tuning and for general model completeness, for several reasons I did not for this network. Firstly, since the goal was to reproduce the Improved U-Net the previously noted paper, this meant there would be no need for the model comparisons which the validation test results can help show. In addition to this the standard parameter selection was already completed by the authors of the paper. Secondly, not having a validation test set meant I could use Keras's lightweight ImageDataGenerator for preprocessing, this greatly simplified pre-processing and processing, however the only way to introduce three way training/validation/testing splits with that generator is to have the data already split into training and testing datasets. The final reason was because of the relatively quick training time and the good results the model achieved straight away meant I had little reason to reduce the size of the training dataset for the creation of a validation set.

The network is trained for 10 epochs using an Adam optimizer with a learning rate of 0.0004.

## Training Results
During training it reached a training Dice Coefficient of 0.8443 and a test Dice Coefficient of 0.9025. The binary accuracy and Dice loss values over the period of the training can be seen below.
![image](https://user-images.githubusercontent.com/14146158/139624335-2c2a7aad-c67d-44e1-a80d-5f03a4b54327.png)
![image](https://user-images.githubusercontent.com/14146158/139624380-53a0b9c5-91b2-4d1a-80cd-93cf13caf535.png)
![image](https://user-images.githubusercontent.com/14146158/139624359-17b0b529-33d0-44ce-8f30-432318a34618.png)

## Visualisation of Results
![image](https://user-images.githubusercontent.com/14146158/139622258-6b6f91cc-259e-4217-ab04-ef73eae4865c.png)
Another set of test images.
![Predict_Images](https://user-images.githubusercontent.com/14146158/139622109-59963ea6-523b-478e-9271-81e7784acb26.png)

## References 
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online].
Available: https://arxiv.org/abs/1802.10508v1
