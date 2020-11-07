
# Segmentation of ISICs dermatology dataset with UNet

This module uses an improved UNet structure to segment the ISICs 2018 dermatological dataset into foreground and background segments. 

#### Dependencies
The following packages are required to run the module correctly
- Tensorflow
- Numpy
- Matplotlib
- Scikit-image
- Scikit-learn
Note that the model relies on Tensorflow only with the remaining packages used to process or display the data.

#### Data Splits
The data was split into portions of 
- Training data (80%)
- Validation data (10%)
- Test data (10%)

These were chosen because the dataset only contains 2594 images and so is not tiny but also not overly large. Cross-validation or other methods such as bootstrapping were not used as they were deemed unnecessary but could be added to boost training size.
