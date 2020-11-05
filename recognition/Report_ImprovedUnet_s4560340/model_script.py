## Importing all the necessary modules from tensorflow
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense,Dropout,UpSampling2D,concatenate, Add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Input


#3Function with the model
def improved_unet_model():

    ## Specifying input size
    input__ = Input((256,256,3))

    ##First conv layer with LeakyRelu activation
    c1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(input__)
    act_1 = LeakyReLU(alpha =0.01)(c1)

    ##first context module with dropout
    context_mod1_layer1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(act_1)
    act_2 = LeakyReLU(alpha =0.01)(context_mod1_layer1)
    cont_dropoutlayer1 = Dropout(.3, input_shape=(2,))(act_2)
    context_mod1_layer2 =  Conv2D(16, (3, 3),  strides=(1, 1), padding='same')(cont_dropoutlayer1)
    act_3 = LeakyReLU(alpha =0.01)(context_mod1_layer2)

    ##element wise addition
    elem_add_1 = Add()([c1, act_3])

    c2 = Conv2D(32, (3, 3),  strides=(2, 2), padding='same')(elem_add_1)
    act_4 = LeakyReLU(alpha =0.01)(c2)

    ##second context module
    context_mod2_layer1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(act_4)
    act_5 = LeakyReLU(alpha =0.01)(context_mod2_layer1)
    cont_dropoutlayer2 = Dropout(.3, input_shape=(2,))(act_5)
    context_mod2_layer2 =  Conv2D(32, (3, 3),  strides=(1, 1), padding='same')(cont_dropoutlayer2)
    act_6 = LeakyReLU(alpha =0.01)(context_mod2_layer2)

    ##element wise add2
    elem_add_2 = Add()([c2, act_6])

    c3 = Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(elem_add_2)
    act_7 = LeakyReLU(alpha =0.01)(c3)

    ##third context module
    context_mod3_layer1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(act_7)
    act_8 = LeakyReLU(alpha =0.01)(context_mod3_layer1)
    cont_dropoutlayer3 = Dropout(.3, input_shape=(2,))(act_8)
    context_mod3_layer2 =  Conv2D(64, (3, 3),  strides=(1, 1), padding='same')(cont_dropoutlayer3)
    act_9 = LeakyReLU(alpha =0.01)(context_mod3_layer2)

    ##element wise add3
    elem_add_3 = Add()([c3, act_9])

    c4 = Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(elem_add_3)
    act_10 = LeakyReLU(alpha =0.01)(c4)

    ##fourth context module
    context_mod4_layer1 = Conv2D(128, (3, 3),  strides=(1, 1), padding='same')(act_10)
    act_11 = LeakyReLU(alpha =0.01)(context_mod4_layer1)
    cont_dropoutlayer4 = Dropout(.3, input_shape=(2,))(act_11)
    context_mod4_layer2 =  Conv2D(128, (3, 3), strides=(1, 1), padding='same')(cont_dropoutlayer4)
    act_12 = LeakyReLU(alpha =0.01)(context_mod4_layer2)

    ##element wise add4
    elem_add_4 = Add()([c4, act_12])

    c5 = Conv2D(256, (3, 3),  strides=(2, 2), padding='same')(elem_add_4)
    act_13 = LeakyReLU(alpha =0.01)(c5)

    ##fifth context module
    context_mod5_layer1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(act_13)
    act_14 = LeakyReLU(alpha =0.01)(context_mod5_layer1)
    cont_dropoutlayer5 = Dropout(.3, input_shape=(2,))(act_14)
    context_mod5_layer2 =  Conv2D(256, (3, 3),  strides=(1, 1), padding='same')(cont_dropoutlayer5)
    act_15 = LeakyReLU(alpha =0.01)(context_mod5_layer2)

    ##element wise add5
    elem_add_5 = Add()([c5, act_15])

    ##upsampling module 1 (in order to be the same size with the corresponding context pathway level for concatenation
    up_1 = UpSampling2D(size=(2, 2))(elem_add_5)
    upconv_1 = Conv2D(128, (3, 3),  strides=(1, 1), padding='same')(up_1)
    act_16 = LeakyReLU(alpha =0.01)(upconv_1)

    #concat 1 
    con1 = concatenate([elem_add_4,act_16], axis = 3)

    ##localization module 1
    loc3x3_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(con1)
    act_17 = LeakyReLU(alpha =0.01)(loc3x3_1)
    loc1x1_1 = Conv2D(128, (1 ,1), strides=(1, 1), padding='same')(act_17)
    act_17 = LeakyReLU(alpha =0.01)(loc1x1_1)

    ##upsampling 2
    up_2 = UpSampling2D(size=(2, 2))(act_17)
    upconv_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(up_2)
    act_18 = LeakyReLU(alpha =0.01)(upconv_2)

    #concat 2
    con2 = concatenate([elem_add_3,act_18], axis = 3)

    ##localization module 2
    loc3x3_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(con2)
    act_19 = LeakyReLU(alpha =0.01)(loc3x3_2)
    loc1x1_2 = Conv2D(64, (1 ,1), strides=(1, 1), padding='same')(act_19)
    act_20 = LeakyReLU(alpha =0.01)(loc1x1_2)

    ##upsampling 3
    up_3 = UpSampling2D(size=(2, 2))(act_20)
    upconv_3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(up_3)
    act_21 = LeakyReLU(alpha =0.01)(upconv_3)

    #concat 3 
    con3 = concatenate([elem_add_2,act_21], axis = 3)

    ##localization module 3
    loc3x3_3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(con3)
    act_22 = LeakyReLU(alpha =0.01)(loc3x3_3)
    loc1x1_3 = Conv2D(32, (1 ,1), strides=(1, 1), padding='same')(loc3x3_3)
    act_23 = LeakyReLU(alpha =0.01)(loc1x1_3)

    ##upsampling 4
    up_4 = UpSampling2D(size=(2, 2))(act_23)
    upconv_4 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(up_4)
    act_24 = LeakyReLU(alpha =0.01)(upconv_4)

    #concat 4
    con4 = concatenate([elem_add_1,act_24], axis = 3)

    c6 = Conv2D(32, (3, 3),  strides=(1, 1), padding='same')(con4)
    act_25 = LeakyReLU(alpha =0.01)(c6)

    ##segmentation layers
    seg_1 = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(c6)
    act_26 = LeakyReLU(alpha =0.01)(seg_1)

    seg_2 = Conv2D(1, (3, 3),  strides=(1, 1), padding='same')(act_23)
    act_27 = LeakyReLU(alpha =0.01)(seg_2)

    seg_3 = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(act_20)
    act_28 = LeakyReLU(alpha =0.01)(seg_3)
    ## Upsampling for addition with second segmentation layer
    up_seg3 = UpSampling2D(size=(2, 2))(act_28)

    ##element wise add6
    elem_add_5 = Add()([act_27, up_seg3])
    ##Upsampling the addition output for addition with first segmentation layer
    up_add5 = UpSampling2D(size=(2, 2))(elem_add_5)

    ##element wise add7
    elem_add_6= Add()([up_add5, act_26])
    ##output layer with sigmoid activation function
    output = Conv2D(1, 1, activation = 'sigmoid')(elem_add_6)

    ## Building model by specifying input and output layers
    model = Model(input__, output)

    return model
