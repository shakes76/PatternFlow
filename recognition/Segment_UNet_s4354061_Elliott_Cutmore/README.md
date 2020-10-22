# Segmentation of ISIC data set using a UNet
This module has three scripts: **segment.py** which holds the 
algorithm specific functions and classes  required for image 
segmentation, **driver_train_save_evaluate.py** and 
**driver_load_evaluate.py** which acts as test scripts 
to run algorithm specific code in a tensorflow GPU 
test environment.

The module created was specifically designed for the segmentation 
of skin irregularities occurring in images from the ISIC data set. 
This segmentation is done using a Convolutional Neural Network 
(**CNN**) structure known as a **UNet**. Coloured images are given
as input images in addition to black and white (binary) segmented masks
corresponding to the input images, known as target images. The CNN maps
the relationship between the input and target images. From this, the UNet
can predict from new test input data (coloured images) what their 
corresponding segmentation mask should look like. The driver scripts 
show example usage of the algorithm specific implementation of the 
segmentation. The **driver_train_save_evaluate.py** script loads 
input and target images, trains a UNet model, saves it to storage then 
evaluates the accuracy of the predicted output segmentation masks of a 
new test set of data. The **driver_load_evaluate.py** script loads in a 
pre-train UNet and evaluates the accuracy of the predicted output 
segmentation masks of a new test set of data. 

A model was trained with the **driver_load_evaluate** script for 100
epochs, 1792 training images and 493 validation images. The average 
dice similarity coefficient    

## segment.py
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
 
## driver_train_save_evaluate.py
This script trains a UNet CNN from the images supplied to it, 
saves the trained UNet model to storage (if given a filename 
to save to) and finally evaluates the model with a test set 
and displays the results. The test set is independent to the 
training and validation sets used to train
the model.

The flow of control is as follows:
1. Load the ISIC image data set filename's
2. Inspect the sizes of the image files corresponding to the 
filename's (if inspect flag set "True")
3. Plot the occurrence of image size's (if inspect flag 
set "True")
4. 
