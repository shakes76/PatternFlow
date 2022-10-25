# 2017 ISIC Challenge Dataset Segmentation with Improved UNet
This algorithm attempts to segment the ISIC Challenge Dataset from 2017 using the Improved 
UNet architecture as described by Isensee et al. [1]. Segmentation in this context involves
converting a high resolution coloured image of a skin lesion into a black and white image
of with a mask highlighting the borders of the skin lesion. This segmentation is the first
step in quantifying the diagnosis of skin lesions as potential melanomas.

# The Improved UNet Algorithm
UNet is a convolutional neural network used for image segmentation that involves encoding,
which condenses an image keeping only the most important information before decoding or upsampling 
this into a higher resolution image which in the case of segmentation is a mask. A feature of
UNet is the propagation of information from the encoding section to the decoding section
through concatenation layers in keras. To ensure shape consistency, the UNet architecture is
therefore a 'U' shape.

Details of the Improved UNet architecture can be found in the paper [1] however in summary,
the primary improvement is the addition of 



# Results
Included in this repository are graphs of dice similarity coefficient (DSC) and loss 
over the course of 50 epochs plotted using matplotlib. There's also a log of each value
at each epoch in training_log.txt. Below is the results of the 50th epoch:
Epoch 50/50
63/63 [==============================] - 31s 498ms/step - loss: 0.0708 - dice_similarity: 0.9224 - val_loss: 0.2983 - val_dice_similarity: 0.7827


# How to Use
As this implementation was coded and tested on Google Colab, it is also the intended
platform for use. .ipynb files have been provided in the repository which can be imported
into Colab. In train.py there are variables store the path for the ISIC data (seg_train_path,
train_path, etc). Note that these paths are google drive paths to ensure easy compatability
with Google Colab so the user should mount their google drive when attempting to run the code
through Colab. If files are loaded locally this may involve slightly altering the code and
importing the python os module.


