# OASIS brain segmentation using Improved UNET
Long Quan Thuan Nguyen - 45104879
## Requirement
 - Python 3.7 (not tested on other version)
 - Tensorflow
 - Matplotlib and IPython for display graph and clear commandline
 - Numpy
## Description
### Image segmentation problem
Image segmentation is a pixel image classification problem where each pixel has its own class. And in OASIS dataset, each pixel has a label by its greyscale intensity. There are 4 greyscales represent each class: 0, 85, 170, and 255. Because the dataset has already seperate training, testing and validating set for us, we don't need to split the data.

### Algorithm
#### Data extraction
For loading the image to memory for deep learning, we can load all of images data into memory. But for this implementation, we will use a custom image generator. Using the generator will limit the amount image data loaded to memory.
In this generator, it will load both the raw image and the mask image from a list of sorted image directories. We need the directories to be sorted for matching the raw image file and the mask image file. We load the image file with grayscale since the pixel class can also work in grayscale. The mask file will be one-hot encoded instead of it's pixel value. We need to extract the classes from one sample image and use the classes to encode the mask image.

#### Model
[**Improved UNET**](https://arxiv.org/abs/1802.10508v1)
![Improved UNET model Image](/examples/imp_unet.png)
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(1, 256, 256, 1)]   0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (1, 256, 256, 16)    160         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (1, 256, 256, 16)    2320        conv2d[0][0]                     
__________________________________________________________________________________________________
dropout (Dropout)               (1, 256, 256, 16)    0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (1, 256, 256, 16)    2320        dropout[0][0]                    
__________________________________________________________________________________________________
add (Add)                       (1, 256, 256, 16)    0           conv2d[0][0]                     
                                                                 conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (1, 128, 128, 32)    4640        add[0][0]                        
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (1, 128, 128, 32)    9248        conv2d_3[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (1, 128, 128, 32)    0           conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (1, 128, 128, 32)    9248        dropout_1[0][0]                  
__________________________________________________________________________________________________
add_1 (Add)                     (1, 128, 128, 32)    0           conv2d_3[0][0]                   
                                                                 conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (1, 64, 64, 64)      18496       add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (1, 64, 64, 64)      36928       conv2d_6[0][0]                   
__________________________________________________________________________________________________
dropout_2 (Dropout)             (1, 64, 64, 64)      0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (1, 64, 64, 64)      36928       dropout_2[0][0]                  
__________________________________________________________________________________________________
add_2 (Add)                     (1, 64, 64, 64)      0           conv2d_6[0][0]                   
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (1, 32, 32, 128)     73856       add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (1, 32, 32, 128)     147584      conv2d_9[0][0]                   
__________________________________________________________________________________________________
dropout_3 (Dropout)             (1, 32, 32, 128)     0           conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (1, 32, 32, 128)     147584      dropout_3[0][0]                  
__________________________________________________________________________________________________
add_3 (Add)                     (1, 32, 32, 128)     0           conv2d_9[0][0]                   
                                                                 conv2d_11[0][0]                  
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (1, 16, 16, 256)     295168      add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (1, 16, 16, 256)     590080      conv2d_12[0][0]                  
__________________________________________________________________________________________________
dropout_4 (Dropout)             (1, 16, 16, 256)     0           conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (1, 16, 16, 256)     590080      dropout_4[0][0]                  
__________________________________________________________________________________________________
add_4 (Add)                     (1, 16, 16, 256)     0           conv2d_12[0][0]                  
                                                                 conv2d_14[0][0]                  
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (1, 32, 32, 256)     0           add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (1, 32, 32, 128)     295040      up_sampling2d[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (1, 32, 32, 256)     0           conv2d_15[0][0]                  
                                                                 add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (1, 32, 32, 128)     295040      concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (1, 32, 32, 128)     16512       conv2d_16[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (1, 64, 64, 128)     0           conv2d_17[0][0]                  
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (1, 64, 64, 64)      73792       up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (1, 64, 64, 128)     0           conv2d_18[0][0]                  
                                                                 add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (1, 64, 64, 64)      73792       concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (1, 64, 64, 64)      4160        conv2d_19[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (1, 128, 128, 64)    0           conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (1, 128, 128, 32)    18464       up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (1, 128, 128, 64)    0           conv2d_21[0][0]                  
                                                                 add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (1, 128, 128, 32)    18464       concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (1, 128, 128, 32)    1056        conv2d_23[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (1, 256, 256, 32)    0           conv2d_24[0][0]                  
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (1, 256, 256, 16)    4624        up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (1, 64, 64, 1)       65          conv2d_20[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (1, 256, 256, 32)    0           conv2d_25[0][0]                  
                                                                 add[0][0]                        
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (1, 128, 128, 1)     33          conv2d_24[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)  (1, 128, 128, 1)     0           conv2d_22[0][0]                  
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (1, 256, 256, 32)    9248        concatenate_3[0][0]              
__________________________________________________________________________________________________
add_5 (Add)                     (1, 128, 128, 1)     0           conv2d_26[0][0]                  
                                                                 up_sampling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (1, 256, 256, 1)     33          conv2d_27[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_5 (UpSampling2D)  (1, 256, 256, 1)     0           add_5[0][0]                      
__________________________________________________________________________________________________
add_6 (Add)                     (1, 256, 256, 1)     0           conv2d_28[0][0]                  
                                                                 up_sampling2d_5[0][0]            
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (1, 256, 256, 4)     8           add_6[0][0]                      
==================================================================================================
Total params: 2,774,971
Trainable params: 2,774,971
Non-trainable params: 0
__________________________________________________________________________________________________
```
There is a slight change from this implementation to the proposed model. We still use `batch normalization` instead of the proposed `instance normalization`.
### Loss and metrics
Since the problem is image segmentation, the metrics and loss function used by this implementation are Dice similarity coefficient and Dice loss. And because OASIS has 4 classes, we need to calculate the average Dice loss for the training. We calculate Dice coefficient for each class and then average them over the number of classes. 

## Example
To run the module, simply run from the command line:
```
python path_to_module/test.py
```
And by default, it will show the model as above and a prediction as follow.
Example image prediction:
![Example prediction](/examples/Example.png)
If you want to train the model, you can change the `EPOCHS` and remove (or rename) the model folder and it will run training session. Also, ensure that you put the OASIS dataset inside `datasets/OASIS` directory on the same level as the `test.py` file.
 - datasets
   - OASIS
     - keras_png_slices_train
     - keras_png_slices_test
     - keras_png_slices_validate
     - keras_png_slices_seg_train
     - keras_png_slices_seg_test
     - keras_png_slices_seg_validate
 - test.py
If using the default model, it can reach dice coefficient score over 0.9 for test set of OASIS dataset.
example evaluation:
`544/544 [==============================] - 10s 18ms/step - loss: 0.0672 - dice_coef_multilabel: 0.9328 - accuracy: 0.9776`