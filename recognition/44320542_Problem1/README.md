# ISICs dataset segmentation with Improved UNet.

### Dependencies
Tensorflow-gpu 2.1
Matplotlib

### Algorithm Description
The aim of this project was to segment the ISICs dataset, which contains 2594 images of lesions and their associated ground truth images. This was done using an Improved UNet model taken from [1].

Firstly, the images are extracted from their directions as both input images and mask (ground truth) images. This data was put into an 80%, 10%, 10% split for training, validation and testing datasets respectively, since it was found that this is a common split precentage. Also, since the dataset is relatively small at ~2500 images, using the majority 80% of the dataset for training is chosen. These images were resized and normalised.

Secondly, the Improved UNet model was implemented, using the information in [1]. Functions to handle context, upsampling and localisation modules were created, with the UNet model implementation falling short in its capabilities. Problems such as sizing issues were encountered when trying to use the tensorflow.keras.tf.Concatenate() function, so unfortunately, UNet model wasn't able to be fully realised. Dice coefficients functions were written, but these can't be tested since the UNet was unable to be implemented.

### Results
Since the Improved UNet couldn't be implemented, results for the whole project weren't attainable. However, outputs from the image loading functions can be seen below in Figure 1.
![](images/LoadImages.JPG)
Figure 1 shows an input image and the corresponding ground truth image.


### References
[1] - F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1

