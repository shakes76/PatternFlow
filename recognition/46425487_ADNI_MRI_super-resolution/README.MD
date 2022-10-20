# ADNI Brain MRI Super-Resolution Network
This folder contains the code to implement, train and test a network to upscale down sampled [ADNI Brain MRI 2D images](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI) by a factor of 4 using the Efficient Sub-Pixel Convolutional Neural Network architecture described in [Wenzhe et al. (2016)](https://arxiv.org/abs/1609.05158). 
## Related Works
Single image super-resolution (SISR) is a much-studied field in image restoration which aims to reconstruct a high-resolution image from its corresponding low-resolution version. Different strategies have been adopted to accomplish this task which include various neural networks such as Super-Resolution Convolutional Neural Network (SRCNN), Fast Super-Resolution Convolutional Neural Network (FSRCNN), and Very Deep Super Resolution (VDSR). However, the stated architectures have drawbacks as they increase the resolution at or before the firs convolution layer. This direct up sampling of the low-resolution image increases the computation complexity. Additionally, interpolation methods used in the up sampling do not help the ill posed nature of the reconstruction problem. 
## ESPCN
Efficient Sub-Pixel Convolutional Neural Network instead perform the upscaling step at the end of a network using an efficient sub-pixel layer. This means that interpolation is no longer needed and the smaller LR representation is directly fed to the network. As a result network is capable of learning a better LR to HR mapping and a smaller filter size is used when learning features which reduces computational complexity. The two steps of learning the feature maps using the convolution and the application of the sub-pixel shuffle function which comprise the ESPCN can be seen in the following diagram: 
![image](https://user-images.githubusercontent.com/69196526/196846707-f431a658-de1f-4197-8e79-fe8f76caea54.png)
*An efficient sub-pixel convolutional neural network [Wenzhe et al. (2016)](https://arxiv.org/abs/1609.05158)*
 

The [model](./modules.py) implemented based on the above architecture can be seen below:

```python
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
## Dependencies
To run the code of this implementation and reproduce the results, the following dependencies are necessary:
* Python 3.9
* Numpy
* Matplotlib
* Tensorflow 2.6 or higher *(You will need tf.keras.utils.image_dataset_from_directory)*
  * *for [GPU support](https://www.tensorflow.org/install/pip) only*  
    * cudatoolkit 
    * cudnn

Additionally, you will need the [ADNI Brain MRI 2D Dataset](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI) as well as to update the paths in the code to import the dataset and store the results of the training

## Preprocessing
No major pre-processing was required to use the [ADNI Brain MRI 2D Dataset](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI). The provided data is already split into train and test categories each with 21500 and 9000 images respectively. The train category was used as the training dataset and test category was used as the validation dataset with exception of a small section (32 images) which were separated to be used for testing and predictions after the training. After this the data image arrays were scaled to be arrays of floating point numbers between 0 and 1. Images were then downsampled by a factor 4 to be used in training and validation of the ESPCN model. 

It's also important to mention that the data was imported as grayscale which eliminated the need for conversions from RGB to YUV formats and separating the Y channel for training and later reconverting the predicted images to RGB which was done in the original ESPCN paper since they dealt with coloured images rather than MRIs. Additionaly, 256x240 images in the dataset were padded to be 256*256 since equal dimensions was required by the implemented model. This was also not a major issue since it only meant adding a few black pixels to the already black background of the MRIs. 

## Results
As it can be seen in the figures below training or validation loss and accuracy did not change much over the course of 50 epochs. However, this is consitent with other ESPCN implementations such as [Long (2020)](https://keras.io/examples/vision/super_resolution_sub_pixel/#introduction)

<img src="https://user-images.githubusercontent.com/69196526/196861403-39c21690-261d-454d-badd-e27ee9d493d3.jpg" width="49%" alt="loss"> <img src="https://user-images.githubusercontent.com/69196526/196861540-735a9f23-3e04-4608-bbed-088a4a82d6f9.jpg" width="49%" alt="accuracy">
*Note: lines are smoothed due to a now fixed bug causing the results to round to 4 decimal places*

Instead [Peak signal-to-noise ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (PSNR) is used as performance metric to study the performance of the netwrok and compare it to other architectures that perform SISR. The ratio indicates the quality difference between an original and a compressed/reconstructed image. The higher the PSNR value the better the closer the compressed/reconstructed image will be to the original. The results below from calculating the PSNR between the low resolution (downsampled) and high resolution (original) versions of test dataset images as well as PSNR between the reconstructed (prediceted by the model) and high resolution images showcases that our model's reconstructed are much closer to their corresponding high resolution image compared to the low resolution image that is fed to the model. 

```
Average PSNR of low res and high res for images is 23.5942
Average PSNR of predicted and high res for images is 28.6388
```

Further quality of reconstructed images by the model support this claim. Below are some examples of images upscaled by the model accomponied by their corresponding low and high resolution versions.

<img src="https://user-images.githubusercontent.com/69196526/196866457-ed322269-7637-4f9f-8486-27cd3871396e.png" width="49%" alt="img8 high res"> <img src="https://user-images.githubusercontent.com/69196526/196866469-33908938-e731-4c80-8481-928097a30c41.png" width="49%" alt="img9 high res">
*High resolution (original) version of the images*

<img src="https://user-images.githubusercontent.com/69196526/196868154-85c5198e-949b-4986-b4d1-886b08a90285.png" width="49%" alt="img8 low res"> <img src="https://user-images.githubusercontent.com/69196526/196869218-878c58eb-f652-4303-a3ae-105c2103e17a.png" width="49%" alt="img9 low res">
*Low resolution (downsampled) version of the images*

<img src="https://user-images.githubusercontent.com/69196526/196868245-22e648ec-2de4-4b9a-af9e-b98a257e3cd0.png" width="49%" alt="img8 predicted res"> <img src="https://user-images.githubusercontent.com/69196526/196869414-3324546a-187d-4f7e-a8f8-f01e65c64b7b.png" width="49%" alt="img9 predicted res">
*Reconstructed version of the images*

