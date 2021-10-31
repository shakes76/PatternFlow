import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, BatchNormalization, Dropout, concatenate, Activation
# Ref: https://github.com/shalabh147/Brain-Tumor-Segmentation-and-Survival-Prediction-using-Deep-Neural-Networks

def doubleConv(input_layer, num_filters, kernel_size=(3,3,3), strides=(1,1,1)):

    X = Conv3D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(input_layer)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(num_filters,kernel_size=kernel_size, strides=strides,padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    return X


def Unet_3d(input_shape=(256,256,128,1), start_neurons=8, dropout=0.2):
    
    input_layer = tf.keras.Input(input_shape)
    # input_layer = tf.image.random_flip_up_down(input_layer, 3)

    # tf.keras.Sequential(
    # [
    #     # tf.image.RandomFlip(mode="horizontal"),
    #     tf.image.random_flip_up_down(image, 3),
    #     tf.image.random_flip_left_right(image, 3),
    # ]
    # )

    c1 = doubleConv(input_layer, start_neurons*1)
    p1 = MaxPool3D(pool_size=(2,2,2), strides=(2,2,2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = doubleConv(p1,start_neurons*2)
    p2 = MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = doubleConv(p2,start_neurons*4)
    p3 = MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = doubleConv(p3,start_neurons*8)
    p4 = MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = doubleConv(p4,start_neurons*16)

    u6 = Conv3DTranspose(start_neurons*8, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(c5)
    u6 = concatenate([u6,c4])
    c6 = doubleConv(u6,start_neurons*8)
    c6 = Dropout(dropout)(c6)
    u7 = Conv3DTranspose(start_neurons*4,kernel_size=(3,3,3), strides=(2,2,2), padding= 'same')(c6)

    u7 = concatenate([u7,c3])
    c7 = doubleConv(u7,start_neurons*4)
    c7 = Dropout(dropout)(c7)
    u8 = Conv3DTranspose(start_neurons*2, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(c7)
    u8 = concatenate([u8,c2])

    c8 = doubleConv(u8,start_neurons*2)
    c8 = Dropout(dropout)(c8)
    u9 = Conv3DTranspose(start_neurons, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(c8)

    u9 = concatenate([u9,c1])
    c9 = doubleConv(u9,start_neurons)

    # 6 is the number of pixel classes in label
    outputs = Conv3D(6, (1, 1,1), activation='softmax')(c9)
    print("########################## output shape: ", outputs.shape)
    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    return model


def dice_coef(y_true, y_pred,  epsilon = 0.00001):
    numerator = 2 * tf.reduce_sum(y_true * y_pred) + epsilon
    denominator = tf.reduce_sum(y_true + y_pred) + epsilon
    return numerator / denominator


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
