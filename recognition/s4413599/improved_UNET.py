from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Conv2D, Input, Dropout, concatenate, Add, Activation, LeakyReLU
from helpFunctions import get_Conv2D, get_contextModule, get_unsamplingModule, get_local_module


# improved unet model
def improved_UNET(shape=(128, 128, 1)):
    '''
    :param shape:
    :return:
    '''
    # Input with input shape
    inputs = Input(shape=shape)
    # 3 * 3 convolution layer with 16 output filters
    convol1 = get_Conv2D(inputs, 16)
    # Context Module with 15 output filters
    context1 = get_contextModule(convol1, 16)
    # Element-wise sum first convolutional layer and context module
    combine1 = Add()([convol1, context1])

    # 3 * 3 convolution layer with 16 output filters with stride 2
    convol2 = get_Conv2D(combine1, 32, strides=(2, 2))
    # Context Module with 32 output filters
    context2 = get_contextModule(convol2, 32)
    # Element-wise sum first convolutional layer and context module
    combine2 = Add()([convol2, context2])

    # 3 * 3 convolution layer with 64 output filters with stride 2
    convol3 = get_Conv2D(combine2, 64, strides=(2, 2))
    # Context Module with 64 output filters
    context3 = get_contextModule(convol3, 64)
    # Element-wise sum first convolutional layer and context module
    combine3 = Add()([convol3, context3])

    # 3 * 3 convolution layer with 128 output filters with stride 2
    convol4 = get_Conv2D(combine3, 128, strides=(2, 2))
    # Context Module with 128 output filters
    context4 = get_contextModule(convol4, 128)
    # Element-wise sum first convolutional layer and context module
    combine4 = Add()([convol4, context4])

    # 3 * 3 convolution layer with 256 output filters with stride 2
    convol5 = get_Conv2D(combine4, 256, strides=(2, 2))
    # Context Module with 256 output filters
    context5 = get_contextModule(convol5, 256)
    # Element-wise sum first convolutional layer and context module
    combine5 = Add()([convol5, context5])
    # Upsampling module with 128 output filters
    upsampling1 = get_unsamplingModule(combine5, 128)

    # Connect first upsampling layer with layer in level 4
    concat4 = concatenate([upsampling1, combine4])
    # Localization layer with 128 output filters
    localize4 = get_local_module(concat4, 128)
    # Upsampling module with 64 output filters
    upsampling2 = get_unsamplingModule(localize4, 64)
    # Connect second upsampling layer with layer on third layer
    concat3 = concatenate([upsampling2, combine3])
    # Localization layer with 64 output filters
    localize3 = get_local_module(concat3, 64)
    # Upsampling layer
    upsampling3 = get_unsamplingModule(localize3, 32)
    # Connect layer with layer on second layer
    concat2 = concatenate([upsampling3, combine2])
    # Get localization module with 32 output filters
    localize2 = get_local_module(concat2, 32)
    # Upsampling module with 16 filters output filters
    upsampling2 = get_unsamplingModule(localize2, 16)

    # Segmentation layer on the layer 3
    segmentation_layer3 = get_Conv2D(localize3, 1, kernel_size=(1, 1), strides=(1, 1))
    # Upscale the segmentation layer
    segmentation_layer3_upscale = UpSampling2D()(segmentation_layer3)
    # Segmentation layer on the layer 2
    segmentation_layer2 = get_Conv2D(localize2, 1, kernel_size=(1, 1), strides=(1, 1))
    # Element-wise sum two segmentation layers
    segmentation_layer3_and_2 = Add()([segmentation_layer3_upscale, segmentation_layer2])
    # Upscale the segmentation layer
    segmentation_layer3_and_2_upscale = UpSampling2D()(segmentation_layer3_and_2)

    # Connect the layer on layer 1
    concat1 = concatenate([upsampling2, combine1])
    # Last convolution layer
    convLast = get_Conv2D(concat1, 32, strides=(1, 1))
    # Get the segmentation layer after the last convolution layer
    segmentation_layer1 = get_Conv2D(convLast, 1, kernel_size=(1, 1), strides=(1, 1))
    # Combine all segmentation layers
    segmentation_combine = Add()([segmentation_layer3_and_2_upscale, segmentation_layer1])
    outputs = Activation('sigmoid')(segmentation_combine)
    return Model(inputs=inputs, outputs=outputs)

# model = improved_UNET()
# print(model.summary())
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

