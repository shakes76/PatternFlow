# Task 4: Imporved Unet for ISIC dataset
## Name: Jiayu Zhou <br> Student No.:45478776
###  Introduction
Unet is a neural network with a " U" shape. It includes Convolution, Max Pooling, Receptive field, batch normalization, Up-sampling, Transposed Convolution, 
Skip Connections and so on. Its activation function is Relu and in the output layer, activation function is Sigmoid or Softmax which is used for multiple labels classification. Compared to other neural network, Unet does not have dense layer, thus its input is able to have various size.<br>
Totally, Unet contains two parts, one is feature extraction and the other one id up sampling, this structure also called as encoder and decoder. 
In feature extraction, image size will be changed because of max pooling. In up sampling, for each up sampling, 
it will merge with feature extraction layer that has the same size. 
Therefore, it works because it combines the location information from the down sampling path with the contextual information in the up sampling path to finally 
obtain a general information combining localisation and context, which is necessary to predict a good segmentation map. It usually be used to find the most significant 
information in images, especially in the field of medical industry. Based on the labels in one image, actually it does the same thing like binary or multiple classes 
classification.  My report is about the improved Unet that can do the segmentation for skin cancer. the model should be trained to classify the different parts in images,
like the background and disease area.<br>
Finally, the improved Unet is based on typical Unet, but the structure has a little bit change. Dice coefficients of test set in model is 0.81.<br>


### Environment: <br>
python==3.7<br> tensorflow==2.3.0 
### Data preparation
There are totally 2594 images, I split them into training set(70%), validation set(20%) and test set(10%). 
For mask images, there are many values in the range of 0 and 255, so I set values that lower than 0.5 as 0 and values that higher than or equal to 0.5 as 1. 
Therefore, there are two labels. Subsequently, I resized the images as (256,256). the input size of model should be (256,256,3) and the output size should be (256,256,1).<br>
train input image example<br>
![image](https://github.com/mollypython/PatternFlow/blob/topic-recognition/recognition/76-report/images/train_input.png)
train label image example<br>
![image](https://github.com/mollypython/PatternFlow/blob/topic-recognition/recognition/76-report/images/train_label.png)<br>

### Improved Unet <br>
There are some differences from Unet. 
The improved Unet includes: first layer-3x3 convolution with 16 filter size and a context module with two 3x3 conv and one dropout layer. Th
ere are totally five similar structures with different filter size in the context path. and then add the first layer and its context module, 
do the same thing for later layers. Subsequently, there is the localization pathway, four up sampling layers in main branch happens here. 
In addition, for each localization module includes a 3x3 conv and 1x1 conv to halves the number of feature maps.
By segmentation layer and upscale in the side branch, and then add them before output, images size increase from 64x64 to 256x256 as the original size. 
Activation function is Relu and in the output layer, the activation function here is Sigmoid because this is a binary classification.
At the same time, merge the information from the down sampling is also important. The initial learning rate is 5e-4 and decay is 10e-5. <br>

### Model sammary<br>
Layer (type)                    Output Shape         Param #     Connected to                     
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 256, 256, 16) 448         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 256, 256, 16) 2320        conv2d[0][0]                     
__________________________________________________________________________________________________
dropout (Dropout)               (None, 256, 256, 16) 0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 256, 256, 16) 2320        dropout[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_AddV2 (TensorFlowOp [(None, 256, 256, 16 0           conv2d_2[0][0]                   
                                                                 conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 128, 128, 16) 0           tf_op_layer_AddV2[0][0]          
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 128, 128, 32) 4640        max_pooling2d[0][0]              
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 128, 128, 32) 0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 128, 128, 32) 9248        dropout_1[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_AddV2_1 (TensorFlow [(None, 128, 128, 32 0           conv2d_4[0][0]                   
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 32)   0           tf_op_layer_AddV2_1[0][0]        
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 64, 64)   18496       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 64, 64, 64)   0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 64, 64, 64)   36928       dropout_2[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_AddV2_2 (TensorFlow [(None, 64, 64, 64)] 0           conv2d_6[0][0]                   
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 64)   0           tf_op_layer_AddV2_2[0][0]        
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 128)  73856       max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 32, 32, 128)  0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 128)  147584      dropout_3[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_AddV2_3 (TensorFlow [(None, 32, 32, 128) 0           conv2d_8[0][0]                   
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 128)  0           tf_op_layer_AddV2_3[0][0]        
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 256)  295168      max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 16, 16, 256)  0           conv2d_9[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 256)  590080      dropout_4[0][0]                  
__________________________________________________________________________________________________
tf_op_layer_AddV2_4 (TensorFlow [(None, 16, 16, 256) 0           conv2d_10[0][0]                  
                                                                 conv2d_10[0][0]                  
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 32, 32, 256)  0           tf_op_layer_AddV2_4[0][0]        
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 32, 32, 384)  0           tf_op_layer_AddV2_3[0][0]        
                                                                 up_sampling2d[0][0]              
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 128)  442496      concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 128)  16512       conv2d_11[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 64, 64, 128)  0           conv2d_12[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 64, 64, 192)  0           tf_op_layer_AddV2_2[0][0]        
                                                                 up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 64, 64, 64)   110656      concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 64, 64, 64)   4160        conv2d_13[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 128, 128, 64) 0           conv2d_14[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 128, 128, 96) 0           tf_op_layer_AddV2_1[0][0]        
                                                                 up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 128, 128, 32) 27680       concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 128, 128, 32) 1056        conv2d_16[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_5 (UpSampling2D)  (None, 256, 256, 32) 0           conv2d_17[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256, 256, 48) 0           tf_op_layer_AddV2[0][0]          
                                                                 up_sampling2d_5[0][0]            
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 256, 256, 32) 13856       concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 256, 256, 1)  33          conv2d_19[0][0]                  
__________________________________________________________________________________________________
Total params: 1,797,537
Trainable params: 1,797,537
Non-trainable params: 0
__________________________________________________________________________________________________<br>
###Dice coefficient<br>
Dice coefficient is two times of the correct prediction pixels divided by the total number of pixels in label image and prediction images. 
It evaluates the similarity of label images and prediction images.<br>


### Training and validation results<br>
Batch size is 2 and epoch is 20.<br>
loss and dice coefficient<br>
![image](https://github.com/mollypython/PatternFlow/blob/topic-recognition/recognition/76-report/images/loss_dice.png)<br>

### Test prediction<br>
![image](https://github.com/mollypython/PatternFlow/blob/topic-recognition/recognition/76-report/images/test_prediction.png)

