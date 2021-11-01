## Perceiver Model Implementation for Classification of the OAI AKOA knee data laterality

# Guide
The code requires the two files, driver.py and model.py to run.
To train the classifier and produce the classifictions for the images, simply download the required files, change the line IMAGE_DIR = '../input/knee-data/AKOA_Analysis/' to the filepath which contains the images and run the driver.py file. 
# Required Imports
The environment must have the following packages to run:
tensorflow==2.6.0
numpy==1.19.5
matplotlib==3.4.3
keras==2.6.0

These are the versions used when running a notebook with Kaggle.

# Perceiver

The Percevier model is a model which builds on transformers while relaxing the assumptions on the relationships between inputs while still being able to scale to a large quantity of inputs. Released by Google Deepmind in June 2021, the model allows for flexibility with the input types and so is used here to classify the laterality of human knees provided by the OAI AKOA Knee Dataset. 

# Data

The OAI AKOA consists of 18680 images from 427 patients with 7760 left knee and 10920 right knee images. The data was processed by reshaping to a size of (73,64) which proved to be successful for other prac students as it saved time while still maintaing accuracy. Furthermore, the images were converted to greysacle and normalized by diviging by 255. 

The data was split into a training set, test set and validation set based on the patient id and not the indiviual images themselves to avoid a patient's images occuring in both the training and validation set themselves which would result in overfitting. 

The split chosen was 0.8:0.04:0.16 for the train:test:validation split.

# Architecture 

**Cross-Attention** 
 First an input latent array and data array are created. The query, key, and value from the data and latent arrays are then passed through and attention and dense layer to resize the output. This output is concatented with the original input, and the result is normalized. Finally, the output is passed through a series of Dense layers to reach the output. 

**Fourier Encoder**
To ensure that the order of the images is maintained, a Fourier Encoding was applied to the inputs. 
First, a logspace between 0 and 6.9 (which was the max/Nyquist Frequency chosen), was initialized and passed into a sine and cosine function. 

This encoding was then applied to the images. 

The Fourier Layer included code from Rishit Dagli, linked here: https://github.com/Rishit-dagli/Perceiver. 


**Transformer**
Following the structure of the paper, the transformer first normalizes the input before passing it through a MultiHeadAttention layer. Then it is projected to the correct size (256,27) using a linear layer where the output is then concatenated with the original input. The output of the concatentaion is normalized again before being passed into a Dense Layer with a GELU activation. This process is repeated four times but can be arbitrarly chosen using a hyper-parameter.  

# Results
**Accuracy Scores Per Epoch**

**Training and Validation Scores**

**Sample Knee Images**

# References
*Rishit Dagli* (26 April 2021). *Perceiver* https://github.com/Rishit-dagli/Perceiver. 

*Khalid Salama* (30 April 2021). *Image classification with Perceiver* https://keras.io/examples/vision/perceiver_image_classification/
