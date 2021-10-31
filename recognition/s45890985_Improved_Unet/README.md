# Segmentation of the ISICs Dataset with Tensorflow Implementation of Improved Unet Model


## Overview of task 
The selected task is to segment the ISICs data set with the Improved Unet [1] model, with a minimum average Dice similarty cofficent (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.8


## Breif Description of the Improved Unet Model
The model architecture of the Unet model is similar to the standard U-net model, comprising of a Encoder path, a bridge, and a decoder path. Forming a U-shaped architecture. However, the improved unet differs in how the layers are connected and extra merging of outputs, using element wise sum, and multiple segmentation layers. In the encoder path, each block consists of a context module, followed by a element wise sum, with blocks connected by a convolution layer of stride 2. In the decoder path, an upsampling module is used, which consits of a simple upsampling followed by a 3x3 convolution layer. There are also an extra segmentation layer after each decoder block, which is upscaled and merged with the final segmentation layer. The segmentation layer is then fed through a softmax layer forming the output. Throughout the model LeakyReLu and InstanceNormalization layers are used after each feature map computing convolution. For a more detailed description of the Improved Unet model refer to [1].


## Algorithm and Results 

### Data Used 
The ISICs data set consists of images of dermatologic related diseases, with a segmentation mask containing 2 segments - 0:Background, 1:Affected Area. The goal of the dataset is to segment the given image using ai for easier dermatologic diagnoistics. In this algorithm the 2018 Challenge dataset for task1-2 (https://challenge.isic-archive.com/data/) is used. This dataset consists of 2594 images in jpg format, and coreesponding mask in png format. The data set is sufficently large to support a train-test split of 80-20, allowing for more images to be used in training while also having enough data for evaluation. The images are also resized to 256x256 to be more easily used by the model. 

### Explanation of the algorithm 
The algorithm implemented is a 2D version of the Improved Unet[1] with minor differences to make it suited to the ISICs data set. The model used follows the same architecture as the improved unet, however instead of Softmax activation, Sigmoid activation is used as there are only 2 classes to segment as opposed to 3 or more.In training, the model follows the paper on improved unet[1], except the loss function of (1-dice coefficent) was also used inplace of a conventional categorical crossentropy loss, as the model is evaluated based on the dice coefficent. 

#### Model.py
Model.py consists of method that deals with each module and layer of the Unet, and a method that creates a model. The model and layers are created using keras. 

#### Test_driver.py
Test_driver.py consists of methods that load data, train the model, and evaluate the model. The data is loaded through tensorflow datasets methods, and are normalized and resized to 256x256. This data is then augmented and fed into the model. The model is then evaluated based on the dice coefficent. 

### Sample results 
The average dice coefficent acheived with 3 epochs was 0.85.A sample display with dice coefficent of 0.96:
![Sample Display](/images/Displayed_sample.png

Loss and accuracy plots: 
![Loss Accuracy plot](/images/Loss_accuracy_plots.png)

### How to use the algorithm
The algorithm could be used by downloading 'the ISIC 2018 task 1' dataset and changing the data directory in driver.py to match the downloaded location. 
example: 
data_path = os.path.join("D:/UQ/2021 Sem 2/COMP3710/Report", "ISIC_2018\ISIC2018_Task1-2_Training_Input")

## Dependencies 
- python 3.9=7
- tensorflow 2.6
- tensorflow-addons 0.14.0
- numpy 1.20.3
- matplotlib 3.4.2


## References 
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online].
Available: https://arxiv.org/abs/1802.10508v1
