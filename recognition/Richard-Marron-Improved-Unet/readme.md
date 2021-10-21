# Skin Lesion Segmentation Using the Improved U-Net

This algorithm uses the improved U-Net model as descibed by [this paper](https://arxiv.org/pdf/1802.10508v1.pdf) by Heidelburg University and Heidelburg Cancer researchers. Below is the network structure as described on page 4 (Fig. 1).

![Network Image](./Figures/Network.png)

## U-Net Structure Details
The name U-Net comes from the shape because the network appears in the shape of the letter "U" and is comprised of two main sections: the down-sampling stage and the up-sampling stage. 
### Down-sampling Stage
Here, we take the image we want to segment and feed it into the network. It then passes through several two-dimensional convolution layers ([Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)) which shrinks the image resolution and increases the number of filters which provides information. Notice in the figure above that there are several connections which jump over the context modules. These help retain some information from before the convolutions and therefore help with performance of the model.

### Up-sampling Stage
After the image has been processed by the down-sampling stage, we move to up-sample the image back to its original resolution. 
