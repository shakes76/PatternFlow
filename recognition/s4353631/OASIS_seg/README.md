## OASIS MRI Segmentation
This folder contains:
* `driver.py` - Driver module which processes data, compiles and trains a model as well as visualizes results
along the way.
* `input_line.py` - Module containing functions for the data input line.
* `layers.py` - Module containing subclassed layers for the model.
* `model.py' - Module containing subclassed model.
* `metrics.py` - Module containing custom metric and loss functions.
* `OASIS_seg.ipynb` - Contents of all modules displayed as a cohesive Jupyter Notebook.

as well as some images of output from `driver.py`. The recognition task I approached was a semantic segmentation 
of the OASIS brain MRI library, with the 3D scans separated into 2D images. As can be seen below, four categories 
were considered including the background. Note that these are 2D MRI slices taken from three separate brains.

![Ground truth input images and segmentation masks.](oasis.png)

In response to this problem, my network architecture of choice was an [https://arxiv.org/abs/1802.10508](improved UNet). 
A standard Unet consists of a convolutional autoencoder with residual connections added between ingoing and outgoing 
layers of the same size feature space. In addition to this underlying frame, my UNet was augmented with 
