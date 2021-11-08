# Improved UNet on ISIC Dataset
###### Molly Parker s4436238 for COMP3710 

### Introduction to UNet

The UNet is a convolutional neural network that gets its name from its shape. The U-shaped network aims to effectively and efficiently segment images and output a segmentation map (i.e. a black and white image showing only the important boundaries from the original images). The downsampling portion of the network works to identify the key segments of the image, and the upsampling portion increases the resolution of the resulting map. 


![UNet architecture by Jeremy Zhang](https://miro.medium.com/max/1838/1*f7YOaE4TWubwaFF7Z1fzNw.png)

### Improved UNet

This model aims to implement the Improved UNet over the ISIC 2018 dataset in order to identify the boundaries of images of skin lesions. The Improved UNet was first created by Isensee et. al. in 2018 for the BraTS (brain tumor segmentation) challenge (https://arxiv.org/pdf/1802.10508v1.pdf). 

![Improved UNet architecture by Isensee et. al.](https://user-images.githubusercontent.com/64070555/140685643-f7781184-baf4-4d88-92bc-eb00d5ecf667.PNG)

### Parameters and Results

I split the ISIC dataset into training, validation, and testing sets for this model at a 6:2:2 ratio. Since the dataset contains images of differing dimensions, I also resized the original images to be 256x256x3, and the segmented images to be 256x256. 


### References

Codella, N., Rotemberg, V., Tschandl, P., Emre Celebi, M., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., Kittler, H., Halpern, A. “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; https://arxiv.org/abs/1902.03368

Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., and Maier-Hein, K.H. “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1

Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015. Springer. Available: https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/

Tschandl, P., Rosendahl, C. and Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).
