from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Concatenate, Conv2DTranspose, Reshape, Permute, Activation, Dropout, Add
from tensorflow.keras.models import Model




#reference unet form demo 2 (scroll down for improved unet code)
def unet(img_height,img_width,num_channels):


    #Input
    inputs = Input((img_height,img_width,1))


    #Downsampling


    layer = Conv2D(64, (3,3), padding="same", activation="relu")(inputs)
    sc1 = Conv2D(64, (3,3), padding="same", activation="relu")(layer)
    mp1 = MaxPooling2D((2, 2))(layer)

    layer = Conv2D(128, (3,3), padding="same", activation="relu")(mp1)
    sc2 = Conv2D(128, (3,3), padding="same", activation="relu")(layer)
    mp2 = MaxPooling2D((2, 2))(layer)

    layer = Conv2D(256, (3,3), padding="same", activation="relu")(mp2)
    sc3 = Conv2D(256, (3,3), padding="same", activation="relu")(layer)
    mp3 = MaxPooling2D((2, 2))(layer)

    layer = Conv2D(512, 3, padding="same", activation="relu")(mp3)
    sc4 = Conv2D(512, 3, padding="same", activation="relu")(layer)
    mp4 = MaxPooling2D((2, 2))(layer)


    #Bottleneck

    layer = Conv2D(1024, (3,3), padding="same", activation="relu")(mp4)
    layer = Conv2D(1024, (3,3), padding="same", activation="relu")(layer)


    #Upsampling with skip connection

    layer = Conv2DTranspose(512, (2, 2), strides=2, padding="same")(layer)
    layer = Concatenate()([layer, sc4])
    layer = Conv2D(512, (3,3), padding="same", activation="relu")(layer)
    layer = Conv2D(512, (3,3), padding="same", activation="relu")(layer)

    layer = Conv2DTranspose(256, (2, 2), strides=2, padding="same")(layer)
    layer = Concatenate()([layer, sc3])
    layer = Conv2D(256, (3,3), padding="same", activation="relu")(layer)
    layer = Conv2D(256, (3,3), padding="same", activation="relu")(layer)

    layer = Conv2DTranspose(128, (2, 2), strides=2, padding="same")(layer)
    layer = Concatenate()([layer, sc2])
    layer = Conv2D(128, (3,3), padding="same", activation="relu")(layer)
    layer = Conv2D(128, (3,3), padding="same", activation="relu")(layer)

    layer = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(layer)
    layer = Concatenate()([layer, sc1])
    layer = Conv2D(64, (3,3), padding="same", activation="relu")(layer)
    layer = Conv2D(64, (3,3), padding="same", activation="relu")(layer)

    #Output

    outputs = Conv2D(num_channels, (1,1), padding="same", activation="softmax")(layer)

    unet = Model(inputs, outputs, name="UNet")

    return unet



#end reference unet form demo 2




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



def unet_improved(img_height,img_width,num_channels):

    #down sample

    inputs = Input((img_height,img_width,1))

    term_1a = Conv2D(16, (3,3), padding="same")(inputs)
    term_1b = context_module(term_1a, 16)
    concat_1a = Add()([term_1a,term_1b])

    term_2a = Conv2D(32, (3,3), strides=2, padding="same")(concat_1a)
    term_2b = context_module(term_2a, 32)
    concat_2a = Add()([term_2a,term_2b])

    term_3a = Conv2D(64, (3,3), strides=2, padding="same")(concat_2a)
    term_3b = context_module(term_3a, 64)
    concat_3a = Add()([term_2a,term_2b])

    term_4a = Conv2D(128, (3,3), strides=2, padding="same")(concat_3a)
    term_4b = context_module(term_4a, 128)
    concat_4a = Add()([term_2a,term_2b])

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

    level_1 = Conv2D(num_channels, (1,1), padding="same")(conv_output)
    level_2 = Conv2D(num_channels, (1,1), padding="same")(local_out_2)
    level_3 = Conv2D(num_channels, (1,1), padding="same")(local_out_3)

    final_term_3 = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(level_3)
    final_term_2 = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(level_2)

    second_last_sum = Add()([final_term_3,final_term_2])
    last_sum = Add()([second_last_sum,level_1])

    output = Activation("sigmoid")(final_term)
   
