# 2017 ISIC Challenge Dataset Segmentation with Improved UNet
This algorithm attempts to segment the ISIC Challenge Dataset from 2017 using the Improved 
UNet architecture as described by Isensee et al. [1]. Segmentation in this context involves
converting a high resolution coloured image of a skin lesion into a bianry mask highlighting 
the borders of the skin lesion. This segmentation is the first step in quantifying the 
diagnosis of skin lesions as potential melanomas.

### The Improved UNet Algorithm
UNet is a convolutional neural network used for image segmentation that involves encoding,
which condenses an image keeping only the most important information before decoding or upsampling 
this into a higher resolution image which in the case of segmentation is a mask. A feature of
UNet is the propagation of information from the encoding section to the decoding section
through concatenation layers in keras. To ensure shape consistency, the UNet architecture is
therefore a 'U' shape.

The improved algorithm works by encoding through a series of context modules and convolutions,
condensing the information to highlight the shape of the lesion. Information is then recovered
through upsampling layers and localisation modules. In addition to the standard encoding to
decoding channels in standard UNet modules, the improved UNet also has segmentation layers
which pass information between the decoding layers helping with classification accuracy. 
Finally a softmax convolution layer produces the output.

### Training and Validation
Training and validation data was simply formatted accordingly with the ISIC dataset, the training
data was used for training and the test data was used for validation. The 'Validation Data' specified
by ISIC can be used for predict.py or for further training/validation if need be.

### Results
Included in this repository are graphs of dice similarity coefficient (DSC) and loss 
over the course of 50 epochs plotted using matplotlib. There's also a log of each value
at each epoch in training_log.txt. Below is the results of the 50th epoch:

Epoch 50/50
63/63 [==============================] - 31s 498ms/step - loss: 0.0708 - dice_similarity: 0.9224 - val_loss: 0.2983 - val_dice_similarity: 0.7827

As seen, the model reaches high dice similarity for the training set but stops just short of the target DSC
of 80 for the validation. It is possible repeated training of this model or a higher number of epochs may
bring this number above 80.


### How to Use
As this implementation was coded and tested on Google Colab, it is also the intended
platform for use. .ipynb files have been provided in the repository which can be imported
into Colab. In train.py there are variables store the path for the ISIC data (seg_train_path,
train_path, etc). Note that these paths are google drive paths to ensure easy compatability
with Google Colab so the user should mount their google drive when attempting to run the code
through Colab. If files are loaded locally this may involve slightly altering the code and
importing the python os module. To initiate training simply run all in train.py.
##### Note:
The downloaded images (i.e. not masks) from ISIC will come with 'superpixels' images which
were manually removed during implementation however dataset.py should account for this by
only loading the jpg files.

### File Structure
/content
    /drive
        /MyDrive
	    /ISIC
		/ISIC-2017_Training_Part1_GroundTruth
		/ISIC-2017_Training_Data
		/ISIC-2017_Test_v2_Part1_GroundTruth
		/ISIC-2017_Test_v2_Data
		/ISIC-2017_Validation_Part1_GroundTruth
		/ISIC-2017_Validation_Data
		/improved_unet.model

### Dependencies
    - Python 3.9
    - tensorflow: Version 2.1
    - random
    - numpy: Version 1.23
    - matplotlib: Version 3.6.1
    - glob
    - cv2: Version 3.4.4

### References
1. https://arxiv.org/abs/1802.10508v1
2. https://en.wikipedia.org/wiki/U-Net
3. https://www.kaggle.com/code/yerramvarun/understanding-dice-coefficient

