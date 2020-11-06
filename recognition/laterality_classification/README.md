# COMP3710 Pattern Recognition Report
#### George Mulhearn (s4532094)

## Purpose
The purpose of this project is to create and assess the performance of a binary
classifier of the AKOA Knee dataset. This dataset contains 18600 png formatted
xray images of left and right knees from 101 unique patients.

## Implementation
The implementation of classification solution is divided into two parts/files;
`laterality_classifier.py` and `driver.py`.
### LateralityClassifier Class
The LateralityClassifier class is what builds the Tensorflow Keras CNN model
capable of binary classification of laterality in these images. The class
has two keras model options, the normal model and an overly simplified model
(for comparision purposes). 

#### Normal Model Architecture
The following architecture was naturally converged towards after 
multiple rounds of testing different structures.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 228, 260, 1)]     0         
_________________________________________________________________
average_pooling2d (AveragePo (None, 57, 65, 1)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 57, 65, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 57, 65, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 28, 32, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 57344)             0         
_________________________________________________________________
dropout (Dropout)            (None, 57344)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               7340160   
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 7,359,105
Trainable params: 7,359,105
Non-trainable params: 0
```
**Note**: *the usage of dropout and exact dropout rate for these dropout layers
is defined in the init parameters of the class.*
#### Simplified Model Architecture
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 228, 260, 1)]     0         
_________________________________________________________________
flatten (Flatten)            (None, 59280)             0         
_________________________________________________________________
dense (Dense)                (None, 1)                 59281     
=================================================================
Total params: 59,281
Trainable params: 59,281
Non-trainable params: 0
```

#### Dependencies
* Tensorflow
* Keras

### Driver
The driver python file is where all dataset processing, usage of the 
LateralityClassifier class and performance analysis takes place.

#### Process
When ran as the main file, the driver does the following:
1. Downloads the AKOA dataset from the cloudstor URL (if it isn't already 
downloaded).
2. Processes the image files, splitting them into training and validation,
 determining the classification label from the file name and ensuring minimal
 data leakage across the sets.*
3. Performs initial dataset inspection: prints ratio of each class, and 
 visualises random images from each set paired with their labels (See Appendix A, B & C).
4. Creates an instance of LateralityClassifier, creates a keras model from it,
 compiles, and trains this model on the processed data.
5. Performance over epochs of training is plotted
6. Calculation of MSE and classification accuracy on the validation set is
 printed
7. Visualisation of predictions and their actual labels is shown for inspection


#####*Solving Data Leakage
This dataset is especially vulnerable to data leakage as there is only 101
unique patient x-rays in entire set, meaning there is roughly 180 images for
each patient, many of which visually appear extremely similar, meaning if the
full set is just randomly shuffled, almost identical x-ray images are bound to
be present in both training and validation sets.. thus leakage.

To mitigate this effect, during the processing of the dataset, patients are
uniquely identified by the filename, and an extra processing step is done to
ensure that the training and validation set do not contain any images from the
same unique patient.

#### Dependencies
* LateralityClassifier (described above)
* Core python libraries (os, random)
* TensorFlow
* Keras
* MatPlotLib
* NumPy

## Results/Output
### Normal Model

### Simplified Model
Although the normal model performed considerably well on the dataset, it may
be over-estimating the difficulty of the problem as using 
LateralityClassifier's simple model usually performs just as well.

#### Example Metric Output
```
mean square error of predictions =  0.02686644094643213
validation acc = 0.9693333333333334
```
#### Example Visualisation Output
![](resources/simple/Model_predictions_on_validation_set.png)
![](resources/simple/Actual_labels_of_validation_set.png)



## Appendix
### Appendix A
![](resources/visualisation_of_testing_set.png)
### Appendix B
![](resources/visualisation_of_training_set.png)
### Appendix C
Example output after initial dataset inspection

```
unique patients in entire dataset:  101
unique patients in training set:  38
unique patients in testing set:  8
number of patients in training and testing:  0

proportion of right knee in training set: 3920 6000
proportion of right knee in test set: 560 1500
```