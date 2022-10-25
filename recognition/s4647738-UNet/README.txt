# 2017 ISIC Challenge Dataset Segmentation with Improved UNet
This algorithm attempts to segment the ISIC Challenge Dataset from 2017 using the Improved 
UNet architecture as described by Isensee et al. [1]. Segmentation in this context involves
converting a high resolution coloured image of a skin lesion into a black and white image
of with a mask highlighting the borders of the skin lesion. This segmentation is the first
step in quantifying the diagnosis of skin lesions as potential melanomas.


# How to Use
As this implementation was coded and tested on Google Colab, it is also the intended
platform for use. .ipynb files have been provided in the repository which can be imported
into Colab. In train.py there are variables store the path for the ISIC data (seg_train_path,
train_path, etc). Note that these paths are google drive paths to ensure easy compatability
with Google Colab so the user should mount their google drive when attempting to run the code
through Colab. If files are loaded locally this may involve slightly altering the code and
importing the python os module.
