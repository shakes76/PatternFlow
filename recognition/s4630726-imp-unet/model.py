from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Concatenate, Conv2DTranspose, Reshape, Permute, Activation, Dropout, Add
from tensorflow.keras.models import Model

#IMPROVED UNET

#individual modules

def context_module(input_layer,filters):
    
    layer = Conv2D(filters, (3,3), padding="same")(input_layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(filters, (3,3), padding="same")(layer)

    return layer

def upsampling_module(input_layer,filters):
    
    layer = Conv2DTranspose(filters, (2,2), strides=2, padding="same")(input_layer)
    layer = Conv2D(filters, (3,3), padding="same")(layer)

    return layer


def localization_module(input_layer,filters):

    layer = Conv2D(filters, (3,3), padding="same")(input_layer)
    layer = Conv2D(filters, (1,1), padding="same")(layer)

    return layer



def unet_improved(img_height,img_width,in_channels,out_channels):

    #down sample

    inputs = Input((img_height,img_width,in_channels))

    term_1a = Conv2D(16, (3,3), padding="same")(inputs)
    term_1b = context_module(term_1a, 16)
    concat_1a = Add()([term_1a,term_1b])

    term_2a = Conv2D(32, (3,3), strides=2, padding="same")(concat_1a)
    term_2b = context_module(term_2a, 32)
    concat_2a = Add()([term_2a,term_2b])

    term_3a = Conv2D(64, (3,3), strides=2, padding="same")(concat_2a)
    term_3b = context_module(term_3a, 64)
    concat_3a = Add()([term_3a,term_3b])

    term_4a = Conv2D(128, (3,3), strides=2, padding="same")(concat_3a)
    term_4b = context_module(term_4a, 128)
    concat_4a = Add()([term_4a,term_4b])

    #Bottle Neck

    term_5a = Conv2D(256, (3,3), strides=2, padding="same")(concat_4a)
    term_5b = context_module(term_5a, 256)
    bottleneck_sum = Add()([term_5a,term_5b])
    concat_4b = upsampling_module(bottleneck_sum, 128)

    #Up sample

    concat_4c = Concatenate()([concat_4a,concat_4b])
    local_out_4 = localization_module(concat_4c, 128)
    concat_3b = upsampling_module(local_out_4, 64)
    

    concat_3c = Concatenate()([concat_3a,concat_3b])
    local_out_3 = localization_module(concat_3c, 64)
    concat_2b = upsampling_module(local_out_3, 32)

    concat_2c = Concatenate()([concat_2a,concat_2b])
    local_out_2 = localization_module(concat_2c, 32)
    concat_1b = upsampling_module(local_out_2, 16)

    concat_1c = Concatenate()([concat_1a,concat_1b])
    conv_output = Conv2D(32, (3,3), padding="same")(concat_1c)

    level_1 = Conv2D(out_channels, (1,1), padding="same")(conv_output)
    level_2 = Conv2D(out_channels, (1,1), padding="same")(local_out_2)
    level_3 = Conv2D(out_channels, (1,1), padding="same")(local_out_3)
    
    final_term_3 = Conv2DTranspose(out_channels, (2, 2), strides=2, padding="same")(level_3)

    second_last_sum = Add()([final_term_3,level_2])

    final_term_2 = Conv2DTranspose(out_channels, (2, 2), strides=2, padding="same")(second_last_sum)

    last_sum = Add()([final_term_2,level_1])

    outputs = Activation("sigmoid")(last_sum)

    unet = Model(inputs, outputs, name="UNet")

    return unet
   
