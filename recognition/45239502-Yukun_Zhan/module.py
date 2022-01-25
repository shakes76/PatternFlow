from tensorflow import keras
import tensorflow_addons as tfa

def context_module(input_layer, filters):
    norm_1 = tfa.layers.InstanceNormalization()(input_layer)
    conv_1 = keras.layers.Conv2D(filters, (3, 3), padding = "same", activation = keras.layers.LeakyReLU(alpha = 0.01))(norm_1)
    drop_layer = keras.layers.Dropout(0.3)(conv_1)
    norm_2 = tfa.layers.InstanceNormalization()(drop_layer)
    conv_2 = keras.layers.Conv2D(filters, (3, 3), padding = "same", activation = keras.layers.LeakyReLU(alpha = 0.01))(norm_2)

    return conv_2

def upsampling_module(input_layer, filters):
    up_layer = keras.layers.UpSampling2D((2, 2))(input_layer)
    up_layer_2 = keras.layers.Conv2D(filters, (3, 3), padding = "same", activation = keras.layers.LeakyReLU(alpha = 0.01))(up_layer)
    norm_1 = tfa.layers.InstanceNormalization()(up_layer_2)
    
    return norm_1

def localization_module(input_layer, filters):
    conv_1 = keras.layers.Conv2D(filters, (3, 3), padding = "same", activation = keras.layers.LeakyReLU(alpha = 0.01))(input_layer)
    norm_1 = tfa.layers.InstanceNormalization()(conv_1)
    conv_2 = keras.layers.Conv2D(filters, (1, 1), padding = "same", activation = keras.layers.LeakyReLU(alpha = 0.01))(norm_1)
    norm_2 = tfa.layers.InstanceNormalization()(conv_2)

    return norm_2

def improved_model():
    input_size = (192, 256, 3)
    
    input_layer = keras.layers.Input(shape=(input_size))
    # Down-sampling
    # layer 1
    # 3x3x3 convolution_1
    conv_1 = keras.layers.Conv2D(16, kernel_size=(3,3), padding="same")(input_layer)
    # context_module_1
    context_module_1 = context_module(conv_1, 16)
    # element-wise sum
    sum_1 = keras.layers.Add()([conv_1, context_module_1])
    
    # layer 2
    conv_2 = keras.layers.Conv2D(32, kernel_size=(3,3), padding="same", strides=(2,2))(sum_1)
    context_module_2 = context_module(conv_2, 32)
    sum_2 = keras.layers.Add()([conv_2, context_module_2])
    
    # layer 3
    conv_3 = keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", strides=(2,2))(sum_2)
    context_module_3 = context_module(conv_3, 64)
    sum_3 = keras.layers.Add()([conv_3, context_module_3])
    
    # layer 4
    conv_4 = keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", strides=(2,2))(sum_3)
    context_module_4 = context_module(conv_4, 128)
    sum_4 = keras.layers.Add()([conv_4, context_module_4])
    
    # layer 5
    conv_5 = keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", strides=(2,2))(sum_4)
    context_module_5 = context_module(conv_5, 256)
    sum_5 = keras.layers.Add()([conv_5, context_module_5])
    
    # localization 
    up_sampling_1 = upsampling_module(sum_5, 128)
    concat_1 = keras.layers.Concatenate()([up_sampling_1, sum_4])
    
    
    # Up-sampling
    # layer 4
    localization_1 = localization_module(concat_1, 128)
    up_sampling_2 = upsampling_module(localization_1, 64)
    
    # layer 3
    concatenation_2 = keras.layers.Concatenate()([up_sampling_2, sum_3])
    localization_2 = localization_module(concatenation_2, 64)
    segmentation_1 = keras.layers.Conv2D(1, (1,1), padding="same", activation=keras.layers.LeakyReLU(alpha=0.01))(localization_2)
    up_sampling_3 = upsampling_module(localization_2, 32)
    
    # layer 2
    concatenation_3 = keras.layers.Concatenate()([up_sampling_3, sum_2])
    localization_3 = localization_module(concatenation_3, 32)
    segmentation_2 = keras.layers.Conv2D(1, (1,1), padding="same", activation=keras.layers.LeakyReLU(alpha=0.01))(localization_3)
    up_sampling_4 = upsampling_module(localization_3, 16)
    up_sampling_segament_1 = keras.layers.UpSampling2D(size=(2,2))(segmentation_1)
    
    sum_sum_1 = keras.layers.Add()([up_sampling_segament_1, segmentation_2])
    
    # layer 1
    concatenation_4 = keras.layers.Concatenate()([up_sampling_4, sum_1])
    conv_6 = keras.layers.Conv2D(32, kernel_size=(3,3), padding="same", activation=keras.layers.LeakyReLU(alpha=0.01))(concatenation_4)
    segmentation_3 = keras.layers.Conv2D(1, (1,1), padding="same", activation=keras.layers.LeakyReLU(alpha=0.01))(conv_6)
    
    up_sampling_segament_2 = keras.layers.UpSampling2D(size=(2,2))(sum_sum_1)
    sum_segmentation = keras.layers.Add()([up_sampling_segament_2, segmentation_3])
    output_layer = keras.layers.Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', padding='same')(sum_segmentation)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model