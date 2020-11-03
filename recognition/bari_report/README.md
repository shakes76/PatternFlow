<h1 align=justify>UNet Model for Image Segmentation</>
  
<h2>Introduction</h2>

### Background

<div style="text-align: justify">The International Skin Imaging Collaboration (ISIC) is a recurring challenge to develop image analysis tools for the automated segmentation and diagnosis of skin lesions with the aim of accurate melanoma detection from dermascopic images. One of three components of this challenge is lesion segmentation.  
Deep convolutional neural network has created a huge bang for last couple of years for its superior performance in image processing and visual recognition tasks. Typically, convolutional neural network-based classification predicts a single class label for an entire image instance. However, certain visual tasks, especially in the domain of the biomedical image processing, require to assign a class label to each pixel of an image that is termed as image segmentation. UNet, which is based on fully connected deep convolutional neural networks (CNN), is widely used for this purpose. A UNet is typically has a contracting path, which is composed of gradually reducing image dimensions through pooling operations but increasing number of feature channels, and an expansion path, which is composed of gradually enhancing image dimension through upsampling operations but decreasing number of image channels. The main idea behind UNet is that with the increased number of feature channels, the contraction path captures the complex image features but loses information on location of those features due to reduction in image dimension. In order to localize, the high resolution features from the contraction path are combined with the upsampled outputs in the expansion path.</div>

### Problem Statement

<div style="text-align: justify">Skin cancer appears to be a major global health concern, as millions of people across the globe are diagnosed with this disease every year. While melanoma is the deadliest form of skin cancer and is responsible for the overwhelming majority of skin cancer related deaths, the survival probability can exceed even 95%, if diagnosed at an early stage. Melanoma can be detection at an early stage by dint of deep CNN based algorithmic image analysis, as pigmented lesions occurring on the surface of the skin. As mentioned earlier, UNet is a handy technique for image segmentation and can be applied for early detection of melanoma by analysing presence of lesion in the skin images. To this end, an UNet model has been designed, trained and evaluated that is described in the following paragraphs.</div>

### Dependencies

**Hardware:** The UNet model developed for this assessment contains dozens of convolutional layers that is very expensive in terms of both computing power and memory allocation and as such, this requires GPU machine to train the model.

**Software:** This model was developed, by and large, based on keras, a high level API of the tensorflow library in python programming language. This has been written and run in python version 3.7. As such, this is recommended to use a version 3.7 or any version released later than that. The libraries used for building this model are listed below:

* python 3.7 (or above)
* tensorflow and its keras API (model building)
* scikit-learn (data partitioning)
* numpy (data preprocessing and post-processing)
* pandas (storing performance metric)
* matplotlib.pyplot (making plot and images)
<p align=center>
  <img src="H:/45420065/lab_report/model_images/25_ep_Tconv/loss_curve.png" width="350" title="hover text">
</p>

<h2>Methods and Algorithm</h2>

### Data Preparation

<div align="justify">The dataset used here was a part ISIC 2018 challenge that consists of 2,594 images in both input and ground truth. All the images have a dimension of 511 by 384 pixel, where all the images in ground truth have the binary masks:
<pre>0: representing the background of the image, or areas outside the primary lesion
255: representing the foreground of the image, or areas inside the primary lesion</code></pre>

First, the dimension of the images was resized to 256 by 256 pixel and all images were converted to grey scale. In order to normalize the data, the pixel values were divided by 255.0 (the highest pixel value). Since the number of available images is not very large, 70% of the images, i.e. 1,816 images were separated for training the model. As the validation set has an important role in tuning model hyperparameters, 20% of total, i.e. 518 images were allocated in validation set. The remaining 10% of total images (260 in number) were kept aside and used exclusively for evaluating the predictive performance of the developed UNet model. The allocation of images among the train, validation and test set was done following the random selection principle.
</div>

### Model Architecture

**Input:** The shape of input tensor is 256x256x1. 

**Contraction Path:** This path consists of 5 repeated applications of two consecutive 3x3 convolutions each followed by Rectified Liner Unit (ReLU) and 2x2 max pooling operations with a stride of 2 for downsampling. At each downsampling step, the number of feature channels doubles. 

**Expansion Path:** Right after the contraction path, this path upsamples the feature map with a 2x2 stride and convolution that halves the number of feature channels that concatenated with the correspondingly cropped feature map from the contracting path to increase the image dimension while reducing the number of feature channels. Then, two 3x3 convolutions, each followed by a ReLU activation, are added at each step in the expansion path. 

**Output:** This consists of 1x1 convolution with a single feature channel that is followed by ‘sigmoid’ activation, since this is a binary classification task. 

Altogether, there are 23 convolutional layers of varied number of feature channels.

**Model Performance**
Category | DSC
---------| -----------
Background | 0.9459
Lesion | 0.8189
Overall | 0.8828
