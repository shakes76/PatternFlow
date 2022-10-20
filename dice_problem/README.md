# UNet segmentation of ISIC
This program uses a UNet neural network to perform segmentation of the ISIC dataset. The ISIC dataset contains over 13,000 dermoscopic images of potential melanoma cases, collected from several leading clinical centers internationally. This program is intended to aid in melanoma diagnosis by segmented images of potential melanoma cases.
The neural network used by this program is a replication of UNet, a convolutional neural network (CNN) for biomedical image segmentation first developed in the Computer Science department at the University of Freiburg.

All data used by this program is avaliable online at https://challenge.isic-archive.com/data/#2017

This program is constructed from the following Python scripts:
* dataset.py responsible for loading all datasets and proprocessing
* modules.py contains the central architecture of the Unet network
* train.py trains the UNet defined in modules.py on the data
* predict.py used to evaluate the trained model