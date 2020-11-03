from tensorflow import keras


def dice_similarity(exp, pred):
    """
    Returns the Dice Similarity Coefficient between an image predicted by the
    model and its expected result.  Uses the formula:
    DSC = 2TP / (2TP + FP + FN)
    Where TP, FP and FN mean "true positive", "false positive" and 
    "false negative", respectively.

    Parameters
    ----------
    exp: numpy array of image data
        Data from the expected image.
    pred: numpy array of image data
        Data from the predicted image.

    Returns
    ----------
    The DSC between the two images.
    """
    # flatten to 1D arrays
    expected = keras.backend.batch_flatten(exp)
    predicted = keras.backend.batch_flatten(pred)
    predicted = keras.backend.round(predicted)

    expected_positive = keras.backend.sum(expected, axis=-1)
    predicted_positive = keras.backend.sum(predicted, axis=-1)

    # TP when both arrays share a positive at the index
    true_positive = keras.backend.sum(expected * predicted, axis=-1)

    # FN is any expected positives not guessed in TP
    false_negative = expected_positive - true_positive

    # FP is any predicted positive not part of TP
    false_positive = predicted_positive - true_positive

    numerator = 2 * true_positive + keras.backend.epsilon()
    denominator = 2 * true_positive + false_positive + false_negative + keras.backend.epsilon()
    return numerator / denominator


def leaky_relu_conv(layer_in, features, stride=1, size=(3,3)):
    conv = keras.layers.Conv2D(features, size, strides=stride, padding="same")(layer_in)
    leaky_relu = keras.layers.LeakyReLU(alpha=0.01)(conv)
    return leaky_relu


def context_module(layer_in, features):
    conv = leaky_relu_conv(layer_in, features)
    conv2 = leaky_relu_conv(conv, features)
    conv2 = keras.layers.Dropout(0.3)(conv2)
    return keras.layers.add([conv2, layer_in])


def upsampling_module(layer_in, features, concat_layer):
    upsamp = keras.layers.UpSampling2D((2, 2))(layer_in)
    conv = leaky_relu_conv(upsamp, features)
    return keras.layers.concatenate([conv, concat_layer])
    

def localisation_module(layer_in, features):
    conv1 = leaky_relu_conv(layer_in, features)
    return leaky_relu_conv(conv1, features, size=(1,1))


def make_model():
    """
    Returns a standard UNET model as per the given lecture slides.

    Has an input shape of (batch_size, 192, 256, 3) and outputs
    one-hot encoded binary images of shape (192, 256, 2).  Uses the Adam
    optimiser, a binary cross-entropy loss function and provides
    dice similarity as a metric.
    """
    input_layer = keras.layers.Input(shape=(192, 256, 3))

    start_conv = leaky_relu_conv(input_layer, 16)

    context1 = context_module(start_conv, 16)
    downconv1 = leaky_relu_conv(context1, 32, stride=2)

    context2 = context_module(downconv1, 32)
    downconv2 = leaky_relu_conv(context2, 64, stride=2)
    
    context3 = context_module(downconv2, 64)
    downconv3 = leaky_relu_conv(context3, 128, stride=2)
    
    context4 = context_module(downconv3, 128)
    downconv4 = leaky_relu_conv(context4, 256, stride=2)

    central_context = context_module(downconv4, 256)
    upconv1 = upsampling_module(central_context, 128, context4)

    local1 = localisation_module(upconv1, 128)
    upconv2 = upsampling_module(local1, 64, context3)

    local2 = localisation_module(upconv2, 64)
    seg1 = leaky_relu_conv(local2, 16, size=(1, 1))
    seg1 = keras.layers.UpSampling2D((2, 2))(seg1)
    upconv3 = upsampling_module(local2, 32, context2)

    local3 = localisation_module(upconv3, 32)
    seg2 = leaky_relu_conv(local3, 16, size=(1,1))
    seg2 = keras.layers.add([seg2, seg1])
    seg2 = keras.layers.UpSampling2D((2, 2))(seg2)
    upconv4 = upsampling_module(local3, 16, context1)

    end_conv = leaky_relu_conv(upconv4, 32)
    seg3 = leaky_relu_conv(end_conv, 16, size=(1, 1))
    seg3 = keras.layers.add([seg3, seg2])
    conv_out = keras.layers.Conv2D(2, (1,1), padding="same", activation="softmax")(seg3)

    model = keras.Model(inputs=input_layer, outputs=conv_out)
    model.compile(optimizer = keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=[dice_similarity])

    return model
    