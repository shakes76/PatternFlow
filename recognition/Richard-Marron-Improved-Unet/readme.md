# Skin Lesion Segmentation Using the Improved U-Net

This algorithm uses the improved U-Net model as descibed by [this paper](https://arxiv.org/pdf/1802.10508v1.pdf) by Heidelburg University and Heidelburg Cancer researchers. 

# Contents Page
* [Setup](#Setup)
    * [Dependencies](#Dependencies)
    * [Usage](#Usage)
* [Network Structure](#U-Net_Structure_Details)
    * [Down-sampling](#Down-sampling_Stage)
    * [Up-sampling](#Up-sampling_Stage)
    * [Model Parameters](#Model_Parameters) 

# Setup
In this section, we will look at how to get this model up and running by using a driver script to communicate with our module.
## Dependencies 
This will be split into two sections, dependencies for the driver script, and another for the module.
### Driver Script
* [numpy](https://numpy.org/install/) - To assist loading and processing data
* [matplotlib](https://matplotlib.org/stable/users/installing.html) - To allow us to see some pretty pictures from the model
* [cv2](https://pypi.org/project/opencv-python/) - Allows us to read in images from disk
* [sklearn](https://scikit-learn.org/stable/install.html) - To split the data into train/test/validation splits
### U-Net Module
* [tensorflow](https://www.tensorflow.org/install) - To build and train the U-Net

Note: It is recommended to use a virtual environment such as one created by [Miniconda](https://docs.conda.io/en/latest/miniconda.html) which allows easy installation of the GPU version of Tensorflow 2 and other dependencies.

## Usage
To run the driver script, simply call `main.py` using Python and it will run for you. Although, by taking a look at the driver code, you will see several options which might be handy if tweaking the model. Firstly, the `main()` function has a parameter called `debugging` which by default is set to `False`. This setting will make the program only process 1/4 of the provided dataset. This is useful if you don't want to sit around and load in the whole dataset every time you run the driver! Other options include the `fit` parameter to the `fit_or_load()` function. This is set to `True` by default which means that the model will be fitted (trained) on the data when the driver runs however, if you already have a trained model, change this setting to `False` and the driver will automatically load your weights and skip the training step.

Inside the `unetModule.py` file, you will see that the Improved U-Net model is defined as a class. This allows for easy access to each of it's paramters as descibed in the [Model Parameters](#Model_Parameters) section.

# U-Net Structure Details
The name U-Net comes from the shape because the network appears in the shape of the letter "U" and is comprised of two main sections: the down-sampling stage and the up-sampling stage. Below is the network structure as described on page 4 (Fig. 1). The goal of this particular implementation is to segment skin lesions from medical images provided by the ISIC 2018 challenge dataset.

![Network Image](./Figures/Network.png)

### Down-sampling Stage
Here, we take the image we want to segment and feed it into the network. It then passes through several two-dimensional convolution layers ([Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)) which shrinks the image resolution and increases the number of filters which provides information. Notice in the figure above that there are several connections which jump over the context modules. These help retain some information from before the convolutions and therefore help with performance of the model.

### Up-sampling Stage
After the image has been processed by the down-sampling stage, we move to up-sample the image back to its original resolution. We introduce up-sampling modules witch apply a transposed two-dimensional convolution ([Conv2DTranspose](https://keras.io/api/layers/convolution_layers/convolution2d_transpose/)) which increases the resolution such that it matches the previous level of the U-Net. 



### Model Parameters
