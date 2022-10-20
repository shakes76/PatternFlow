# UNet segmentation of ISIC
This program uses a UNet neural network to perform segmentation of the ISIC dataset. 

All data used by this program is avaliable online at https://challenge.isic-archive.com/data/#2017

This program is constructed from the following Python scripts:
* dataset.py responsible for loading all datasets and proprocessing
* modules.py contains the central architecture of the Unet network
* train.py trains the UNet defined in modules.py on the data
* predict.py used to evaluate the trained model