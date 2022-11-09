# Brain MRI Semantic Image Segmentation using improved U-net
## Introduction
For this project, I have built an improved Unet and performed image segmentation on the ISIC dataset, which are images of skin cancer. In this documentation, I will introduce the structure of the model, model training process and its results based on the model. 

## Model Architecture 
### Improved Unet 
A Unet is a convolutional neural network which uses a U-shaped encoder-decoder. The main idea is to supplement a usual contracting network by successive layers, where pooling operations are replaced by upsampling operators. Hence these layers increase the resolution of the output. What's more, a successive convolutional layer can then learn to assemble a precise output based on this information.

This improved Unet comprises a context aggregation pathway that encodes increasingly abstract representations of the input as we progress
deeper into the network, followed by a localization pathway that recombines
these representations with shallower features to precisely localize the structures
of interest.

![Improved Unet model](/images/unet_model.png)

Figure above is the network architecture of the improved unet. The activations in the context pathway are computed by context modules.
Each context module is in fact a pre-activation residual block with two
3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between. Context
modules are connected by 3x3x3 convolutions with input stride 2 to reduce the
resolution of the feature maps and allow for more features while descending down
the aggregation pathway. 

The localization pathway is designed to take features
from lower levels of the network that encode contextual information at low spatial resolution and transfer that information to a higher spatial resolution. This
is achieved by first upsampling the low resolution feature maps, which is done by
means of a simple upscale that repeats the feature voxels twice in each spatial
dimension, followed by a 3x3x3 convolution that halves the number of feature
maps.

We then recombine the upsampled features with the features from the corresponding level of the context aggregation
pathway via concatenation. Following the concatenation, a localization module
recombines these features together. It also further reduces the number of feature
maps which is critical for reducing memory consumption. A localization module
consists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the
number of feature maps.

Throughout the network we use leaky ReLU nonlinearities with a negative slope of 10−2
for all feature map computing convolutions. We furthermore replace the traditional
batch with instance normalization since we found that the stochasticity
induced by our small batch sizes may destabilize batch normalization.

### Training procedure
Using the ISIC dataset of the split ratio 60:20:20 for the training, validation and testing dataset, the network is trained with randomly sampled patches of size 256*192*3 voxels and batch size 2. This is done so that the model is not overfitting on the training set and the test set gets a consistent representation sample of the whole dataset. The model is then trained for a total of 15 epochs. The training is done using the adam optimizer with a learning rate of 0.0001.

## Usage

The driver script is set up with all required calls in its main routine. With the below dependencies available simply call "python main.py". The ISIC data must be in the path data/ISIC2018_... relative to main.py.

### Dependencies
Beyond standard python libraries, the model and its driver script require the following packages:
* Tensorflow 2.5 (Pre-processing and training model)
* Matplotlib (Display results)

## Example Output
### Training Behaviour 
![Dice coefficient](/images/dice_coiff.png)

The average Dice Coefficient over the test set is: 0.90

### Output comparison
![Predict1](/images/predicted_img.png)
![Predict2](/images/predicted_img2.png)
![Predict3](/images/predicted_img3.png)