<h1 align=center>UNet Model for Image Segmentation</h1>
  
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

**Inputs for Driver Script:** The following inputs will be required from the user for running the driver script:
* Full path of the source directory for input data
* Full path of the source directory for ground truth
* Number of epochs to be used for training the model

<h2>Methods and Algorithm</h2>

### Data Partition

<div style="text-align: justify">The dataset used here was a part ISIC 2018 challenge that consists of 2,594 images in both input and ground truth. All the images have a dimension of 511 by 384 pixel, where all the images in ground truth have the binary masks:</div>
  
<pre>0: representing the background of the image, or areas outside the primary lesion
255: representing the foreground of the image, or areas inside the primary lesion</pre>

<div style="text-align: justify">At first, the dimension of the images was resized to 256 by 256 pixel and all images were converted to grey scale. In order to normalize the data, the pixel values were divided by 255.0 (the highest pixel value). Since the number of available images is not very large, 70% of the images, i.e. 1,816 images were separated for training the model. As the validation set has an important role in tuning model hyperparameters, 20% of total, i.e. 518 images were allocated in validation set. The remaining 10% of total images (260 in number) were kept aside and used exclusively for evaluating the predictive performance of the developed UNet model. The allocation of images among the train, validation and test set was done following the random selection principle.</div>

### Model Architecture

**Input:** The shape of input tensor is 256x256x1. 

**Contraction Path:** This path consists of 5 repeated applications of two consecutive 3x3 convolutions each followed by Rectified Liner Unit (ReLU) and 2x2 max pooling operations with a stride of 2 for downsampling. At each downsampling step, the number of feature channels doubles. 

**Expansion Path:** Right after the contraction path, this path upsamples the feature map with a 2x2 stride and convolution that halves the number of feature channels that concatenated with the correspondingly cropped feature map from the contracting path to increase the image dimension while reducing the number of feature channels. Then, two 3x3 convolutions, each followed by a ReLU activation, are added at each step in the expansion path. 

**Output:** This consists of 1x1 convolution with a single feature channel that is followed by ‘sigmoid’ activation, since this is a binary classification task. 

Altogether, there are 23 convolutional layers of varied number of feature channels.

### Model Compilation and Training

<div style="text-align: justify">The model is compiled with Adam optimizer, as it’s the ability to adapt the learning rate during the course of training the model and is very handy in reaching local and global minima very fast. Besides, ‘binary_crossentropy’ loss was chosen since this is a binary classification problem. In the course of model fitting, callbacks were applied to save best model along with parameter weights, which would be loaded once the training was over. Though the model is expected to be trained in 50 epochs, the training might stop early if loss doesn’t improve in ten consecutive epochs even after lowering the learning rates after 5 epochs. The learning curve showing how training and validation loss changes with increased number of epochs is appended below:</div>

<p align=center>
<img src="https://github.com/s4542006/PatternFlow/blob/topic-recognition/recognition/bari_report/image/loss_curve.png" width="750" title="learning curve">
</p>
<div style="text-align: justify">This is obvious from the above curve that both training and validation loss curves gradually decline with the number of epochs increases and eventually start to become flat just before reaching the fortieth epoch. Another important observation from the plot is that the deviation between training and validation curves has all the way remained in a narrow corridor hinting a very limited overfitting problem of the model trained here.</div>

### Classification Threshold

<div style="text-align: justify">Since the model is trained having a single feature channel (without one hot encoding) in both training and validation set the ground truths, the model prediction provides a single probability value per instance. Hence, setting a threshold is essential to distinguish between two prediction classes, e.g. background and lesion (foreground). For this particular problem, the thresh was set at 0.5, i.e. any predicted probability above 0.50 was categorized as lesion and any predicted probability equal to or below the threshold was categorized as background.</div>

<h2>Discussion and Conclusion</h2>

### Model Performance Analysis

<div style="text-align: justify">As mentioned earlier, a separate test set was used for evaluating the model’s predictive performance. For assessing the model performance with the test set, each pixel’s predicted mask (label) is compared with that in the ground truth. There are many techniques available for this sort of comparison. One of the most widely used techniques is Dice Similarity Coefficient (DSC), which can be determined by twice intersection divided by union (very similar to F1-score). For getting the class-wise DSC, both ground truth image and predicted image masks (labels) were one hot encoded and then, the aforesaid method was applied. Both class-wise and overall DSC of the built U-net are appended below:
</div>


Category | DSC
---------| -----------
Background | 0.9447
Lesion | 0.8183
Overall | 0.8789


Couple of predicted (segmented) images in grey scale along with corresponding images in input and ground truth of the test set are appended below.

<p align = center>
<img src="https://github.com/s4542006/PatternFlow/blob/topic-recognition/recognition/bari_report/image/sample_predicted_image_1.png" width="900" title="sample test images">
<img src="https://github.com/s4542006/PatternFlow/blob/topic-recognition/recognition/bari_report/image/sample_predicted_image_2.png" width="900" title="sample test images">
</p>

### Conclusion
<div>The performance of the built U-net model is quite impressive, although there is room for improvement. Such improvement can be achieved either by tweaking the model architecture or augmenting the input images or even by enriching the dataset by adding more images along with properly masked ground truth.</div>
