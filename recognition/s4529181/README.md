# segment the ISICs data set with the Improved UNet

# author: Yaoyu Liu

# student number: 45291818

# generalization
UNET is a convolutional neural network. ISIC is a dataset of skin cancer photos. I need to use the improved UNET model to segment ISIC. The model I created can predict with a dice similarity coefficient of 0.81.

This model is used to refer to the preprocessing version taught by teacher 3710, and also to UNET on the TF official website. https://www.tensorflow.org/tutorials/images/segmentation?hl=zh

# algorithm description
UNET is an encoder decoder structure,
This structure is to convolute and pool the pictures first. In my UNET, the pictures are pooled four times. At the beginning, the pictures are 256 * 256, which will become four features of different sizes: 128 * 128, 64 * 64, 32 * 32 and 16 * 16. Then we do up sampling or deconvolution on the smallest feature image. After four up sampling, we can get a 256 * 256 result with the same size as the input image.
<p align="center">
  <img width="700" src="img/unet_model"/>
</p>

# input data
To run this program, you need to download ISIC 2018 dataset, https://challenge2018.isic-archive.com/

Change the path in the text to the path where your data set is located
images = glob.glob(
'C:\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2')
labels = glob.glob(
'C:\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2')

## Dependencies
 - Python 3.8.10
 - Tensorflow 2.1.0
 - Numpy 1.19.1
 - Matplotlib 3.3.1
install pip
