# AKOA Knee Laterality Classification with Perceiver Transformer

![img.png](img.png)

Perceiver Transformer Architecture:
-
The Perceiver Transformer is a type of transformer network which relies on cross-attention layers
instead of self-attention layers. This helps reduce the dimensionality of the model such that

The Osteoarthritis Initiative Accelerated Knee Osteroarthritis (OAI AKOA) dataset contains 
~19000 MRI images of patients left and right knees with and without osteoarthritis. Building classifiers
on such a dataset is important in the field of medical research as the onset of this disease is 
poorly understood.

The Perceiver model constructed in the current report has the following structure:

Latent Array Size: 128



Requirements:
- 
- Tensorflow version >= 2.4.1
- tensorflow-addons
- tensorflow-datasets
- scikit-learn
- matplotlib
- numpy

Instructions:
-
1. Place labelled AKOA dataset in the project directory.
2. Create conda environment and install requirements:
   - conda create -n <ENV_NAME> tensorflow-gpu
   - pip install requirements
    
3. Run: driver.py <NAME_OF_AKOA_DATASET_DIRECTORY>
    - on the first run this will sort images into the 'datasets' directory  

Training:
-

The images in the AKOA dataset are of size 260 x 228, which are downsampled to 16 x 16
for the training and validation of the model, with grayscale normalisation (1./127.5 - 127.5). 
This allowed fast training performance as well as strong inference due to the reduction of
overfitting observed in longer training experiments (50 epochs) with larger image sizes.
Since the classification problem is relatively simple, detecting whether the image is of a 
left or right knee, it is very prone to overfitting, hence why this highly downsampled training set
likely performs well. The input data is split into 75:25 training/test, with a large portion
of test data chosen to again reduce overfitting in the training data and give an accurate
representation of performance upon model evaluation. The training data was split into 80:20 
training and validation, which was chosen as a fairly routine split ratio for machine learning
experiments. Overall this training procedure produced excellent results, as observed below.

![input_train_image.png](input_train_image.png)

Results
-
![test_acc.png](test_acc.png)

![Successful_Training_1.png](Successful_Training_1.png)

We observe in the given images of test accuracy and the training and validation plot below it
that the training was highly convergent in a matter of 10 epochs to above the 90% threshold, 
with a high test set accuracy of 97.16%.

References:
-
Model based on:
https://keras.io/examples/vision/perceiver_image_classification/