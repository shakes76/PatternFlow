# Comp-3710: Final Report

## Topic chosen

>Create a generative model of the OASIS brain data set using a DCGAN.

## Written by

> Shashank(45771767)


## Objective 

> To generate fake images of OASIS brain dataset images, which are simialr to original images in the dataset using DCGAN.

## Constraint

The fake and original image should have a structural similarity of atleast 0.6

## Generative Adverserial networks:

These are generator-discriminator based model, which are used to generate like-pixel images using the training data.


## Modules:

### libraries.py

This module contains all the required libraries to run this code.

### check-gpu.py 

This module is used to check to whether your enviroment is using GPU or not.

### helper_functions.py 

This module contains all the required helper functions to run the core algorithm of GAN.
This module contains the following functions:

    1. generator function:
        Required inputs : img_shape(X),noise_shape(100)

        Expected output : random image of size (X*X*3) 

    2.build_discriminator:
        Required_inputs: img_shape(X*X*3)

        Expected output: single value(probability)

    3. combiner:
        Required inputs: noise_shape(100), instance of generator, instance of discriminator.

    4. get_noise:
        Required_inputs: size of the nosie.

    5. plot_generated _images:
        Required_inputs: Generated noise from noise function, instance of generator, n_sample: number of sample images to be generated.

### main.py 

Module used to store the core algorithm of DACGAN. This function conatins the follwoing functions.
    
    1. get_npdata: Used to reszie and rescale the original images.
        required input: size of the training dataset.

    2. train function: Used the train the DCGAN.
        required inputs: 
            1. models : Load the instances of combiner, discriminator and generator in the follwoign order.

            2. X_train: Load the training data output from get_npdata.

            3. epochs (deafault value is 10)

            4. batch_size ( default value is 128)

        Expected output:
            1. Generator and discrimnator losses will be generated.

        Working of train function(Core algorithm of GAN):

        1. Train the discriminator amd compute the loss based on it.

        2. Freeze the layers of discriminator and train the combiner based on the loss calculated by the discriminator.

        3. Train the discriminator and compute the loss again.

        4. Repeat the above three steps until a reasonable image is generated.

### load_model.py 

Used to load the trained model and generate random fake images which will hopefully be similar to the original images.

### metric_check.py 

Used to check the structural similarity between the fake and the original images.


## Usage:

### Requried libraries:

1. matplotlib
2. os 
3. time  
4. numpy 
5. tensorflow
6. PIL 
7. tqdm

please make sure you install all these libraries using pip or conda.

## How to run the code?

> step 1: Once you have installed all the libraries, run check-gpu.py to checl whether your enivronment is using GPU. if not please install tensorflow-gpu.

>step 2: Go to the main function and change the dir_data variable to appropriate location and if you want to change the shape of the fake image generated, please change the img_shape variable to appropriate size.(At the moment only 64,128,256 sizes are avaiable in the generator function)

>step 3: Go to the end of main.py module and change the model saving directory to appropriate location based on your desktop. the recommendation would be save the trained model in saved_models directory.

>step 4: Run the main.py function and you can see that your model will be saved in saved_folders directory(generator64.h5),where 64 represents the size of the image. we can also the plots of generator and discrimiantor loss.

>step 5: Open the load_model.py. the first thing to do is changing the directory in the load_model function to an appropriate place and generate fake images.

>step 6: Once you are done with saving the model, open the metric_check.py file and change the directories to suitable locations and to check the structural similarity between the fake and the original images.










