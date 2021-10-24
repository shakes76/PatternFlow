from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Concatenate, Conv2DTranspose, Reshape, Permute, Activation
from tensorflow.keras.models import Model

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
