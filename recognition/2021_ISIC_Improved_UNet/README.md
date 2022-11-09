# Segmentation of the ISICs dataset using Improved UNet
Published by the International Skin Imaging Collaboration (ISIC), the ISICs 2018 dataset
was used as part of the Skin Lesion Analysis Towards Melanoma Detection recurring challenge.
This challenge is an opportunity for participants to develop automated tools which aid in the
diagnosis of melanoma. This project employs the Improved UNet [1] architecture to perform
segmentation on the aforementioned ISICs dataset. 

## ISICs dataset
The ISIC's dataset contains 2594 images and their associated segmentation masks. By default, the dataset contains images of varying sizes and aspect ratios. Data normalisation and resizing was applied to provide consistency throughout the inputs.

<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/ExampleISIC.jpg">
</p>

<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/ExampleISIC_Segmentation.png">
</p>

_Figure 1: Sample image and associated mask from the ISICs 2018 dataset_

### Data preprocessing
As part of the pre-processing phase, all of the images (training images and masks) were normalised. In order to be run through the network, all of the images had to be the same size. The size chosen was (192, 256). The training images kept 3 colour channels: [192, 256, 3]. On the other hand, the segmentation masks were reduced to a single colour channel: [192, 256, 1]. The segmentation masks were also thresholded: pixels with a value > 0.5 after normalisation were set to 1, and the rest were set to 0. 

#### Training, Test & Validation Split.
The Training, Testing and Validation data split chosen was 70 / 15 / 15. Some research was conducted on the optimal split for medical data. In general, it was found that there is no single correct split, however this percentage seemed to be the most highly regarded. For a dataset of this size, that means there was around 1800 training samples, and 390 training & validation samples.  

## Architecture
Proposed in 2018 [1], the Improved UNet is designed upon the original model of UNet, proposed in 2015 [2]. 

<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/ImprovedUNetArchitecture.png">
</p>

_Figure 2: Improved UNet Architecture [1]_

The Improved UNet is composed of two main sections, the Context Aggregation pathway and the Localisation pathway. These pathways share information about the input images through Skip Connections from the Context Aggregation Pathway.

### Context Modules & The Context Aggregation Pathway
The Context Aggregation pathway is designed to encode the input images into increasingly compact representations as the network progresses. To do so, it is composed of a collection of 3x3 Convolutions (with a stride of 2) and Context Modules.

The layer-by-layer architecture of the Context Modules is as follows:

|Context Module Architecture|
|-|
|Instance Normalization|
|Leaky ReLU Activation|
|3x3 Convolution|
|Dropout (_0.3 dropout rate_)|
|Instance Normalization|
|Leaky ReLU Activation|
|3x3 Convolution|

### Localisation Modules & The Localisation Pathway
The Localisation Pathway is designed to increase the dimensionality of the encoded image representation to produce high resolution segmentations by means of Localisation Modules, UpSampling modules and image upscaling.

The layer-by-layer architecture of the Localisation Modules is as follows:

|Localisation Module Architecture|
|-|
|3x3 Convolution|
|1x1 Convolution|

#### Up-Sampling Modules
Up-Sampling modules are placed after every localisation module in the Localisation Pathway. 

The layer-by-layer architecture of the Up-Sampling Modules is as follows:

|Up-Sampling Module Architecture|
|-|
|2D UpSampling layer (2x2)|
|3x3 Convolution|

### Skip Connections
Denoted by the horizontal dashed lines in _Figure 2_, Skip Connections are element-wise summations of the 3x3 (stride 2) Convolutions and Context Module outputs' in the Context Aggregation pathway. Skip Connections are concatenated into the corresponding network level in the Localisation Pathway. 

The Localisation Modules are designed to re-introduce these skip connections into the network after the concatenation. 

### Segmentation
Segmentation occurs 3 times in the Localisation Pathway. Performing segmentation on multiple levels of the network allows for segmented information from the lower levels to propagate into the higher levels through an element-wise summation.

Segmentation layers are 3x3 convolutions with a single output filter.

The 'U' shaped dashed lines in _Figure 2_ denote the pathway that the segmentation levels take. Output is taken from the levels' Localisation Module and given to a Segmentation Layer. Lower (smaller) layers are up-sampled to allow element-wise summation to occur. 

## Optimizer & Loss
The optimizer used in this implementation was the [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) with a learning rate of 5e-4, as per [1].

### Dice Similarity Coefficient
The [Dice Similarity Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) is a common metric used in segmentation problems. Formally, DSC is defined as:

<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/DiceCoefficient.png">
</p>

That is, the DSC is: 2 * the overlap between the pixels in the Ground Truth segmentation mask, and the model-generated Segmentation Mask. This is then divided by the sum of the total pixels in both masks. 

## Results
For the following results, the model was run for 20 epochs. The `validation` performance follows the `training` performance closely throughout training, divering slightly towards the end. The distribution of DSC values on the Test Set evaluation is left-skewed, with few images attaining low DSC scores. Overall, an average DSC of > 0.8 was attained on the Test Set, with > 67% of images yielding a DSC score of 0.8 or higher.

### Accuracy & Loss Plots
The following plots show the behaviour of the model in terms of DSC and Loss over a 20 epoch run.

<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/Figure_1.png">
</p>

_Figure 3: Improved UNet model loss_


<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/Figure_2.png">
</p>

_Figure 4: Improved UNet model dice coefficient_

### Performance on the Test Set
After the model was trained for 20 epochs, its performance was evaluated on the Test Set.

<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/modelMetrics.png">
</p>

_Figure 5: Model performance on the Test Set after training_

From _Figure 5_, we see that the average Dice Coefficient was 80.8%. Overall, 67.8% of the test set yielded a DSC of 0.8+.

#### DSC Distribution
The histogram below shows the distribution of the Test Set's DSC values during evaluation.


<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/histogram.png">
</p>

_Figure 6: Distribution of DSC on the Test Set evaluation_

### Output generated
Masks output by the model were thresholded such that pixels which were > 0.5 were set to 1, else they were set to 0. Below are some output examples from the trained model, on the test set.

<p align="center">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/Figure_5.png">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/Figure_6.png">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/Figure_10.png">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/Figure_20.png">
  <img src="https://github.com/default-jamc/PatternFlow/blob/topic-recognition/recognition/2021_ISIC_Improved_UNet/images/Figure_22.png">
</p>

_Figure 7: Input image / Ground Truth mask / Model-generated mask_

## Additions and Changes
The architecture described above gives an overview of the design of the model.
During development, it was found that making slight tweaks to the architecture resulted in better performance. These changes were:
- `UpSampling2D` layers used the `interpolation='bilinear'` parameter as opposed to the default `interpolation='nearest'`

## Usage
To run this network, ensure you have the appropriate Dependencies installed. 

Download the ISIC's 2018 dataset and place the training images and segmentation masks in two separate folders in the directory where the `model.py` and `driver.py` are located, named as so:
- Training images: ISIC2018_Task1-2_Training_Input_x2 
- Segmentation masks: ISIC2018_Task1_Training_GroundTruth_x2

Open up a commandline and navigate to the directory where `driver.py` is saved, and run it:

`python driver.py`

To ensure the data is loaded correctly, the Training Input from _Figure 1_ should appear on-screen, followed by its corresponding mask from the Training GroundTruth. 

You may change the amount of epochs that the network runs for and the `Adam` learning rate by changing the variables at the top of `driver.py`

- `EPOCHS` denotes the total amount of epochs.
- `OPT_LEARNING_RATE` denotes the `Adam` learning rate.

Once the network is finished, 
1. It will generate `Loss` and `Dice Coefficient` graphs as shown in _Figure 3_ and _Figure 4_ above. 
2. It will then proceed to evaluate the test set, and some performance metrics will be output to the screen, as shown in _Figure 5_ above.
3. A histogram of the Test Set's DSC distribution throughout evaluation will be generated, as shown in _Figure 6_ above.
4. 20 images of the Original Image / Ground Truth Mask / Model-generated Mask will be generated, as shown in _Figure 7_ above. (_Note: you may change the amount of images output using the `local_batch` variable in the `generatePredictions` method in `driver.py`_)



## Dependencies
- Python 3.9.6
- Tensorflow 2.6.0
- Matplotlib 3.4.2
- Numpy 1.19.5
- Tensorflow Addons 0.14.0

## References
[1]: Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., Maier-Hein, K.H, "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge". _arXiv: Computer Vision and Pattern Recognition_, 2018.

[2]: Ronneberger, O., Fischer, P., Brox, T., "U-net: Convolutional networks for biomedical image segmentation,". _International Conference on Medical Image Computing and Computer-Assisted Intervention_, 2015. (Springer, pp. 234-241).
