# ISIC data set with U-Net
My solution to the ISIC data set using a U-Net model.

The folder contains the following two files:
* driver_script.py
* solution.py


## solution.py

This is the U-Net model. It is implemented entirely in TensorFlow.

This file does not need to be run. Instead, it is imported into driver_script.py


## driver_script.py

This is the driver script. This file:
* imports the data.
* manipulates the data into various datasets for training, validating and testing.
* import the model from solution.py and compile this model.
* train the model using the datasets.
* makes and plots predictions using the model.

Call this file to run.
Will need to change where your images are saved. Currently, I am pulling them from my computer.