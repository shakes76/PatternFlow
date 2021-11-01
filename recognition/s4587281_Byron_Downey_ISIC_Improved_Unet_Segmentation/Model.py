from tensorflow.keras.layers import Input, Conv2D, Dropout, LeakyReLU, BatchNormalization, UpSampling2D, concatenate, Add, Dense
from tensorflow.keras import Model

def create_model(num_epochs, batch_size, input_shape, base_channels, dropout_rate, leaky_relu_slope, kernel_size, upsampling_kernel_size):
    #Defining Model Structure with Keras layers
    
    input = Input(shape=input_shape)
    
    #first downscaling section (16 channels by default)
    initial_conv = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (input)
    context1_conv1 = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (initial_conv)
    context1_dropout = Dropout(dropout_rate) (context1_conv1)
    context1_conv2 = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context1_dropout)
    context1_batch_norm = BatchNormalization() (context1_conv2)
    
    #second downscaling section (32 channels by default)
    stride2_conv1 = Conv2D(base_channels * 2, (2,2), kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context1_batch_norm)
    context2_conv1 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv1)
    context2_dropout = Dropout(dropout_rate) (context2_conv1)
    context2_conv2 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context2_dropout)
    context2_batch_norm = BatchNormalization() (context2_conv2)
    
    #third downscaling section (64 channels by default)
    stride2_conv2 = Conv2D(base_channels * 4, (2,2), kernel_size, padding = "same",activation=LeakyReLU(leaky_relu_slope)) (context2_batch_norm)
    context3_conv1 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv2)
    context3_dropout = Dropout(dropout_rate) (context3_conv1)
    context3_conv2 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context3_dropout)
    context3_batch_norm = BatchNormalization() (context3_conv2)
    
    #fourth downscaling section (128 channels by default)
    stride2_conv3 = Conv2D(base_channels * 8, (2,2), kernel_size, padding = "same",activation=LeakyReLU(leaky_relu_slope)) (context3_batch_norm)
    context4_conv1 = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv3)
    context4_dropout = Dropout(dropout_rate) (context4_conv1)
    context4_conv2 = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context4_dropout)
    context4_batch_norm = BatchNormalization() (context4_conv2)
    
    #bottom of "U" shape of the Unet - (256 channels by default)
    stride2_conv4 = Conv2D(base_channels * 16, (2,2), kernel_size, padding = "same",activation=LeakyReLU(leaky_relu_slope)) (context4_batch_norm)
    context5_conv1 = Conv2D(base_channels * 16, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv4)
    context5_dropout = Dropout(dropout_rate) (context5_conv1)
    context5_conv2 = Conv2D(base_channels * 16, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context5_dropout)
    context5_batch_norm = BatchNormalization() (context5_conv2)
    
    #first upsampling module
    upsample1__upsample_layer = UpSampling2D(upsampling_kernel_size) (context5_batch_norm)
    upsample1_conv = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample1__upsample_layer)
    
    concat1 = concatenate([context4_batch_norm, upsample1_conv])
    
    #first localisation module
    localisation1_conv1 = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat1)
    localisation1_conv2 = Conv2D(base_channels * 8, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (localisation1_conv1)
    
    #second upsampling module
    upsample2_upsample_layer = UpSampling2D(upsampling_kernel_size) (localisation1_conv2)
    upsample2_conv = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample2_upsample_layer)
    
    concat2 = concatenate([context3_batch_norm, upsample2_conv])
    
    #second localisation module
    localisation2_conv1 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat2)
    localisation2_conv2 = Conv2D(base_channels * 4, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (localisation2_conv1)
    
    #third upsampling module
    upsample3_upsample_layer = UpSampling2D(upsampling_kernel_size) (localisation2_conv2)
    upsample3_conv = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample3_upsample_layer)
    
    concat3 = concatenate([context2_batch_norm, upsample3_conv])
    
    #third localisation module
    localisation3_conv1 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat3)
    localisation3_conv2 = Conv2D(base_channels * 2, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (localisation3_conv1)
    
    #fourth upsampling module
    upsample4_upsample_layer = UpSampling2D(upsampling_kernel_size) (localisation3_conv2)
    upsample4_conv = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample4_upsample_layer)
    
    #final section
    concat4 = concatenate([context1_batch_norm, upsample4_conv])
    final_conv = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat4)
    
    segmentation_layer1 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (localisation2_conv2)
    
    #upsamples previous segmentation layer so it has the same dimensions and can be added with the later one
    segmentation_layer1_upsampled = UpSampling2D(upsampling_kernel_size) (segmentation_layer1)
    segmentation_layer2 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (localisation3_conv2)
    
    #adds the segmentation layers and upsamples the result so it can be added with the last one
    add1 = Add() ([segmentation_layer1_upsampled, segmentation_layer2])
    add1_upsampled = UpSampling2D(upsampling_kernel_size) (add1)
    
    segmentation_layer3 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (final_conv)
    
    #adds the final segmentation layer to the previous 2
    add2 = Add() ([add1_upsampled, segmentation_layer3])
    
    output = Conv2D(1, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (add2)
    #sigmoid activation function, as there are only 2 classes represented by 0 and 1
    output = Dense(3, activation="sigmoid") (add2)

    unet = Model(input, output)
    return unet
    
