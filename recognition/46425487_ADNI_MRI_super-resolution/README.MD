# ADNI Brain MRI Super-Resolution Network
This folder contains the code to implement, train and test a network to upscale down sampled ADNI Brain MRI 2D image data by a factor of 4 using the Efficient Sub-Pixel Convolutional Neural Network architecture described in [Wenzhe et al. (2016)](https://arxiv.org/abs/1609.05158). 
## Related Works
Single image super-resolution (SISR) is a much-studied field in image restoration which aims to reconstruct a high-resolution image from its corresponding low-resolution version. Different strategies have been adopted to accomplish this task which include various neural networks such as Super-Resolution Convolutional Neural Network (SRCNN), Fast Super-Resolution Convolutional Neural Network (FSRCNN), and Very Deep Super Resolution (VDSR). However, the stated architectures have drawbacks as they increase the resolution at or before the firs convolution layer. This direct up sampling of the low-resolution image increases the computation complexity. Additionally, interpolation methods used in the up sampling do not help the ill posed nature of the reconstruction problem. 
## ESPCN
Efficient Sub-Pixel Convolutional Neural Network instead perform the upscaling step at the end of a network using an efficient sub-pixel layer. This means that interpolation is no longer needed and the smaller LR representation is directly fed to the network. As a result network is capable of learning a better LR to HR mapping and a smaller filter size is used when learning features which reduces computational complexity. The two steps of learning the feature maps using the convolution and the application of the sub-pixel shuffle function which comprise the ESPCN can be seen in the following diagram: 
![image](https://user-images.githubusercontent.com/69196526/196846707-f431a658-de1f-4197-8e79-fe8f76caea54.png)
*An efficient sub-pixel convolutional neural network [Wenzhe et al. (2016)](https://arxiv.org/abs/1609.05158)*


The [model](./modules.py) implemented based on the above architecture can be seen below:

```
def get_model(upscale_factor=4, channels=1):
    """
    This function creates the model to up scales images by the given upscale factor
    """
    inputs = keras.Input(shape=(None, None, 1))
    x = layers.Conv2D(64, 5, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(inputs)
    x = layers.Conv2D(64, 3, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(x)
    x = layers.Conv2D(32, 3, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
```
