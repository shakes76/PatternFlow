import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

def improved_unet_a(input_size=(256, 256, 3),output_channels=2):
    
    #16 output filters 3*3 convolution layer
    inputs = Input(input_size)
    conv1 = Conv2D(16, (3, 3), padding="same")(inputs)
    av1 =LeakyReLU(alpha=0.01)(conv1)

    #16 output filters context module
    context16 = Conv2D(16, (3, 3), padding="same" )(av1) 
    context16 = BatchNormalization()(context16)
    context16 = LeakyReLU(alpha=0.01)(context16)
    context16 = Dropout(0.3) (context16)
    context16 = Conv2D(16, (3, 3), padding="same")(context16)
    context16 = BatchNormalization()(context16)
    context16 = LeakyReLU(alpha=0.01)(context16)
    #16 element-wise sum
    add16 = Add()([av1,context16])
    
    #down sampling
    #32 output filters 3*3 convolution layer 
    dp32 = Conv2D(32, (3, 3), strides=(2,2), padding="same")(add16)
    dp_av32 = LeakyReLU(alpha=0.01)(dp32)   
    #32 output filters context module
    context32 = Conv2D(32, (3, 3), padding="same" )(dp_av32) 
    context32 = BatchNormalization()(context32)
    context32 = LeakyReLU(alpha=0.01)(context32)
    context32 = Dropout(0.3) (context32)
    context32 = Conv2D(32, (3, 3), padding="same")(context32)
    context32 = BatchNormalization()(context32)
    context32 = LeakyReLU(alpha=0.01)(context32)
    #32 element-wise sum
    add32=Add()([dp_av32,context32])

    #64 output filters 3*3 convolution layer 
    dp64 = Conv2D(64, (3, 3), strides=(2,2), padding="same")(add32)
    dp_av64 = LeakyReLU(alpha=0.01)(dp64)
    
    #64 output filters 3*3 context module 
    context64 = Conv2D(64, (3, 3), padding="same" )(dp_av64) 
    context64 = BatchNormalization()(context64)
    context64 = LeakyReLU(alpha=0.01)(context64)    
    context64 = Dropout(0.3) (context64)
    context64 = Conv2D(64, (3, 3), padding="same")(context64)
    context64 = BatchNormalization()(context64)
    context64 = LeakyReLU(alpha=0.01)(context64)

    #64 element-wise sum
    add64=Add()([dp_av64,context64])    
    
    
    
    #128 output filters 3*3 convolution layer
    dp128 = Conv2D(128, (3, 3), strides=(2,2), padding="same")(add64)
    dp_av128 = LeakyReLU(alpha=0.01)(dp128)
    
    #128 output filters 3*3 context module 
    context128 = Conv2D(128, (3, 3), padding="same" )(dp_av128) 
    context128 = BatchNormalization()(dp_av128)
    context128 = LeakyReLU(alpha=0.01)(context128)
    context128 = Dropout(0.3) (context128)
    context128 = Conv2D(128, (3, 3), padding="same")(context128)
    context128 = BatchNormalization()(context128)
    context128 = LeakyReLU(alpha=0.01)(context128)

    #128 element-wise sum
    add128=Add()([dp_av128,context128])
    
    ###256 output filters 3*3 convolution layer
    dp256= Conv2D(256, (3, 3), strides=(2,2), padding="same")(add128)
    dp_av256 = LeakyReLU(alpha=0.01)(dp256)
    
    ###256 output filters 3*3 context module
    context256 = Conv2D(256, (3, 3), padding="same" )(dp_av256)
    context256 = BatchNormalization()(dp_av256)
    context256 = LeakyReLU(alpha=0.01)(context256) 
    context256 = Dropout(0.3) (context256)
    context256 = Conv2D(256, (3, 3), padding="same")(context256)
    context256 = BatchNormalization()(context256)
    context256 = LeakyReLU(alpha=0.01)(context256)
    #256 element-wise sum
    add256 = Add()([dp_av256,context256])
    
    
    ####128 output filters upsamplling module
    up128 = UpSampling2D()(add256)
    up128 = Conv2D(128, (3, 3), padding="same")(up128)
    up128 = LeakyReLU(alpha=0.01)(up128)
    ## 128 skip connection
    skip128= Concatenate()([up128, add128])    
    ###128 output filters localization module
    
    loc128 = Conv2D(128, (3, 3) , padding="same")(skip128)
    loc128 = BatchNormalization()(loc128)
    loc128 = LeakyReLU(alpha=0.01)(loc128)
    loc128 = Conv2D(128, (1, 1), padding="same")(loc128)
    loc128 = BatchNormalization()(loc128)
    loc128 = LeakyReLU(alpha=0.01)(loc128)


    
    
    ####64 output filters upsamplling module
    up64 = UpSampling2D()(loc128)
    up64 = Conv2D(64, (3, 3), padding="same")(up64)
    up64= LeakyReLU(alpha=0.01)(up64)
    #64 skip connection
    skip64= Concatenate()([up64, add64])
    
    ###64 output filters localization module
    loc64 = Conv2D(64, (3, 3) , padding="same")(skip64)
    loc64 = BatchNormalization()(loc64)
    loc64 = LeakyReLU(alpha=0.01)(loc64)
    loc64 = Conv2D(64, (1, 1), padding="same")(loc64)
    loc64 = BatchNormalization()(loc64)
    loc64 = LeakyReLU(alpha=0.01)(loc64)
    
    
    ##64 output filters segmentation layer
    


        
    seg64 = Conv2D(output_channels, (3, 3), padding="same")(loc64)
    seg64 = LeakyReLU(alpha=0.01)(seg64)
       
    
    ##32 output filters upsamplling module
    up32 = UpSampling2D()(loc64)
    up32 = Conv2D(32, (3, 3), padding="same")(up32)
    up32= LeakyReLU(alpha=0.01)(up32) 
    
    
    #32 skip connection
    skip32= Concatenate()([up32, add32])    
    
    ###32 output filters localization module
    loc32 = Conv2D(32, (3, 3) , padding="same")(skip32)
    loc32 = BatchNormalization()(loc32)
    loc32 = LeakyReLU(alpha=0.01)(loc32)
    loc32 = Conv2D(32, (1, 1), padding="same")(loc32)
    loc32 = BatchNormalization()(loc32)
    loc32 = LeakyReLU(alpha=0.01)(loc32)
    

    #32 output filters segmentation layer
    seg32 = Conv2D(output_channels, (3, 3), padding="same")(loc32)
    seg32 = LeakyReLU(alpha=0.01)(seg32)

    
    
    #16 output filters upsamplling module
    up16 = UpSampling2D()(loc32)
    up16 = Conv2D(16, (3, 3), padding="same")(up16)
    up16= LeakyReLU(alpha=0.01)(up16)   
    ##16 skip connection
    skip16= Concatenate()([up16, add16])


            
    #32 output filters 2D convolution
    conv32_out = Conv2D(32, (3, 3), padding="same")(skip16)
    conv32_out = LeakyReLU(alpha=0.01)(conv32_out)    
    #32 output filters segmentation layer
    seg32_final = Conv2D(2, (3, 3), padding="same")(conv32_out)
    seg32_final = LeakyReLU(alpha=0.01)(seg32_final)    
    ## 64 up sampling
    seg64_up_sampling =UpSampling2D()(seg64)
    ## element_wise sum
    add_seg = Add()([seg64_up_sampling ,seg32])    
    ## 32 up sampling
    seg32_up_sampling = UpSampling2D()(add_seg)
    ## element_wise sum
    element_wise_sum = Add()([seg32_final , seg32_up_sampling])
    #out put layer throung sigmoid
    output = Conv2D(output_channels, (3, 3), activation="sigmoid", padding="same")(element_wise_sum)
    model = Model(inputs=inputs, outputs=output)
    return model

    
    
    
    
    
  
