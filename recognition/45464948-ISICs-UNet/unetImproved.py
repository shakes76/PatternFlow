from tensorflow.keras.layers import BatchNormalization , ReLU ,Dropout
from tensorflow.keras.layers import Input , Conv2D, Add
from tensorflow.keras.layers import UpSampling2D , concatenate, LeakyReLU
from tensorflow.keras.models import Model
import tensorflow as tf

def contextModel(input_layer,conv):
    # from easy, each context module is in fact a pre-activation residual block with two
    # 3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between
    block = BatchNormalization()(input_layer)
    block = ReLU()(block)
    block = Conv2D(conv, (3, 3), padding='same')(block)
    # dropout layer (pdrop = 0.3) in between
    block = Dropout(0.3)(block)
    block = BatchNormalization()(block)
    block = ReLU()(block)
    block = Conv2D(conv, (3, 3), padding='same')(block)
    return block


def unetmodel():

    #encoder part
    input_layer = Input(shape=(256,256,3))
    #convolution
    conv1 = Conv2D(16,(3,3), padding='same')(input_layer)
    # context module
    contextModel1 = contextModel(conv1,16)
    #element wise sum
    ews1 = Add()([conv1,contextModel1])# ews = element wise sum
    #convolution stride2
    conv2 = Conv2D(32, (3, 3), strides=(2,2), padding='same')(ews1)
    # context module
    contextModel2 = contextModel(conv2, 32)
    # element wise sum
    ews2 = Add()([conv2, contextModel2])  # ews = element wise sum
    # convolution stride2
    conv3 = Conv2D(64, (3, 3), strides=(2,2), padding='same')(ews2)
    # context module
    contextModel3 = contextModel(conv3, 64)
    # element wise sum
    ews3 = Add()([conv3, contextModel3])  # ews = element wise sum
    # convolution stride2
    conv4 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(ews3)
    # context module
    contextModel4 = contextModel(conv4, 128)
    # element wise sum
    ews4 = Add()([conv4, contextModel4])  # ews = element wise sum
    # convolution stride2
    conv5 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(ews4)
    # context module
    contextModel5 = contextModel(conv5, 256)
    # element wise sum
    ews5 = Add()([conv5, contextModel5])  # ews = element wise sum

    #decoder part
    # upsampling module
    up_layer6 = UpSampling2D((2, 2))(ews5)
    up_layer6 = Conv2D(128, (3, 3), activation= LeakyReLU(alpha= 0.01), padding='same')(up_layer6)
    # concatenate
    conc6 = concatenate([ews4, up_layer6])
    # localization module
    loca6 = Conv2D(128, (3, 3), activation= LeakyReLU(alpha= 0.01), padding='same')(conc6)
    loca6 = BatchNormalization()(loca6)
    loca6 = Conv2D(128, (1, 1), activation= LeakyReLU(alpha= 0.01),padding='same')(loca6)
    loca6 = BatchNormalization()(loca6)
    # upsampling module
    up_layer7 = UpSampling2D((2, 2))(loca6)
    up_layer7 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(up_layer7)
    # concatenate
    conc7 = concatenate([ews3, up_layer7])
    # localization module
    loca7 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conc7)
    loca7 = BatchNormalization()(loca7)
    loca7 = Conv2D(64, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(loca7)
    loca7 = BatchNormalization()(loca7)
    #segmentation layer
    seg7 = Conv2D(1,(1,1))(loca7)
    seg7 = UpSampling2D((2,2))(seg7)
    # upsampling module
    up_layer8 = UpSampling2D((2, 2))(loca7)
    up_layer8 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(up_layer8)
    # concatenate
    conc8 = concatenate([ews2, up_layer8])
    # localization module
    loca8 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conc8)
    loca8 = BatchNormalization()(loca8)
    loca8 = Conv2D(32, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(loca8)
    loca8 = BatchNormalization()(loca8)
    # segmentation layer
    seg8 = Conv2D(1, (1, 1))(loca8)
    seg8 = Add()([seg7, seg8])
    seg8 = UpSampling2D((2, 2))(seg8)
    # upsampling module
    up_layer9 = UpSampling2D((2, 2))(loca8)
    up_layer9 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(up_layer9)
    # concatenate
    conc9 = concatenate([ews1, up_layer9])
    #convolution
    conv9 = Conv2D(32, (3,3), activation = LeakyReLU(alpha=0.01), padding='same')(conc9)
    #segmentation layer
    seg9 = Conv2D(1, (1, 1))(conv9)
    seg9 = Add()([seg9, seg8])

    output_layer = Conv2D(1, (1,1),activation='sigmoid')(seg9)


    unetmodel = Model(input_layer,output_layer)
    return unetmodel

def fit(model,x,y, epoch_size, batch):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                     metrics=['accuracy'])

    model.fit(x, y, epochs=epoch_size, batch_size=batch,
                    validation_split=0.2)