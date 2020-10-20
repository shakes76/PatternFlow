# Segmentation of ISIC data set using a UNet
This module has two scripts: **segment.py** which holds the 
algorithm specific functions and classes and **driver.py** 
which acts as a test script to run algorithm specific code 
from segment.py

The module created was specifically designed for the segmentation 
task of skin irregularities occurring in the ISIC data set. 
This segmentation is done using a Convolutional Neural Network 
(**CNN**) structure known as a **UNet**. Coloured input images are  

## Segment.py
The **segment.py** module created can: 
* Load images from storage into RAM or image generators,
* Inpect images to determine occurrence of appearing 
image sizes,
* Split single data set into training, validation and 
test data set's,
* Create image generators compatable with 
**tensorflow.keras.Model** methods for training as some 
data sets do not fit completely into RAM,
* Create a typical UNet CNN model for training, prediction
 and evalutions of image data, and
* Display the **Dice Similarity Coefficients** of target
 segmentation images and their trained model prediction 
 output counter-parts. 
 
## Driver.py