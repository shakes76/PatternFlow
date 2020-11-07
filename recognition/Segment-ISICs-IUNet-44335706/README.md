
# Segmentation of ISICs dermatology dataset with UNet

This module uses an improved UNet structure to segment the ISICs 2018 dermatological dataset into foreground and background segments. 

### Dependencies
The following packages are required to run the module correctly
- Tensorflow
- Numpy
- Matplotlib
- Scikit-image
- Scikit-learn
Note that the model relies on Tensorflow only with the remaining packages used to process or display the data.

### Usage


### Methodology
An explanation of the inner-workings of the model and the techiques employed.

#### Data Processing
The images were all resized to 256x256 to give them a uniform standard shape well suited to a convolutional neural net.

**Original image and segmentation**

<img src="resources/ISIC_0000256.jpg" height="300"> <img src="resources/ISIC_0000256_segmentation.png" height="300">

**Resized image and segmentation**

![Resized input image](resources/ISIC_000256_resized.jpg) ![Resized segmented image](resources/ISIC_000256_segmentation_resized.png)

These images were provided to the model using a custom Keras Sequence class named iunet_sequence that provides the data in batches and avoided loading too many images into memory at once. These images were fed into the model structure as can be seen below:

MODEL STRUCTURE IMAGE HERE

#### Data Splits
The data was split into portions of 
- Training data (80%)
- Validation data (10%)
- Test data (10%)

These were chosen because the dataset only contains 2594 images and so is not tiny but also not overly large. Cross-validation or other methods such as bootstrapping were not used as they were deemed unnecessary but could be added to boost training size.
