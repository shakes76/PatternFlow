# ISIC 2017 Skin lesion segementation using an improved U-Net

## Dataset
ISIC2017 dataset can be found on the following link

ISIC2017 dataset : https://challenge.isic-archive.com/data/#2017

## Description of the algorithm
![KakaoTalk_Snapshot_20221021_175510](https://user-images.githubusercontent.com/59554674/197143729-01160b28-8c62-4da2-b7b7-4b9041676450.png)

We applied the network architecture as seen in the image above. The network architecture was referenced from 'Brain Tumor Segmentation and Radiomics
Survival Prediction: Contribution to the BRATS 2017 Challenge' by Isensee, et al. The link will be shared below.

Basically, this improved U-net is based on the existing U-net architectures. However, its core architecure inside the network is rather diffrent to the others.
And the diffrences are that it has context, localization, and upsampling modules:

Context module consists of: Two 3 x 3 convolutional layers and a dropout layer with rate 0.3.
Localization module consists of: A 3 x 3 convolutional layer and a 1 x 1 convolutional layer. 
Upsampling modules consists of: An upsampling layer and a 3 x 3 convolutional layer.

In this architecture, there are two pathways, which are the context pathway(left) and the localization pathway(right).
Through the context pathway, the U-net reduces the resolution of the feature maps by using the context module.
Through the localiztion pathway, the U-net take features for the lower level to the higher level by using the localization module that upsampling the low resolution feature maps. After this, the concatenation between the upsampled features and the context aggregation in the same level is done.

Finally, the output forms via element-wise sum between the concatenation above and the segmentation layers.

reference : https://arxiv.org/abs/1802.10508v1

## Task

Train a model for the image segmentation of skin lesion using the improved U-net architecture.

## How it works

### Data preprocessing
Firstly, we needed to resize all the images to 256 x 256 because the images of the dataset did not have the same size. And saved all the resized images into new directories. Then, created blank arrays to store the resized training, validation, and test images and joined the resized images to the blank arrays respectively.
This process has taken some time. And created data frames to store the excel files containing the names of the images. After that, loaded and generated the normalized images and masks of training, validation, and test data.

### Training
After data preprocessing, we had train_x, train_y, validation_x, and validation_y. And loaded an improve U-net model that takes an image with 256 x 256 x 3 created by Input fucntion in Keras library. And we set the learning rate as 0.0005 and the decay rate as learning rate x 0.985 as the paper stated. And we compiled a model using Adam for optimization, and dice similarity loss wss used as a metric. And we fitted the model with train_x, train_y, validation_x, and validation_y. The batch size was 8, and epochs was 300.

### Results

We achieved a dice similarity of 0.51.

reference : https://arxiv.org/pdf/1606.04797v1.pdf

![KakaoTalk_Snapshot_20221021_224451](https://user-images.githubusercontent.com/59554674/197198882-7c2e081a-90f5-4db8-9886-19d771081ea4.png)

The images below are good cases and bad cases.
#### Good results
![Result_1](https://user-images.githubusercontent.com/59554674/197197864-a2404488-0146-4190-8e3e-35ded7f01f7f.png)
![Result2](https://user-images.githubusercontent.com/59554674/197197901-ed236ae2-3b32-4811-8a2a-bc9b1f16d160.png)
![Result3](https://user-images.githubusercontent.com/59554674/197197930-1021a9c5-31b7-4f7a-927a-2a42e103a797.png)

#### Bad results
![BadResult](https://user-images.githubusercontent.com/59554674/197198195-f4304b79-409e-40ef-bf39-ff10fe5f41a8.png)
![BadResult2](https://user-images.githubusercontent.com/59554674/197198811-41bb423e-4ad1-45af-bef6-54296ea50c1b.png)

## Environment
OS : Windows10
Python : 3.7
Cuda : 10.1
Tensorflow: 2.1.0


