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


def make_model():
    """
    Returns a standard UNET model as per the given lecture slides.

    Has an input shape of (batch_size, 192, 256, 3) and outputs
    one-hot encoded binary images of shape (192, 256, 2).  Uses the Adam
    optimiser, a binary cross-entropy loss function and provides
    dice similarity as a metric.
    """
    input_layer = keras.layers.Input(shape=(192, 256, 3))

    conv1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = keras.layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = keras.layers.MaxPooling2D((2, 2))(conv4)

    conv_mid = keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv_mid = keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv_mid)
    upsamp1 = keras.layers.UpSampling2D((2, 2))(conv_mid)

    upconv1 = keras.layers.concatenate([upsamp1, conv4])
    upconv1 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(upconv1)
    upconv1 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(upconv1)
    upsamp2 = keras.layers.UpSampling2D((2, 2))(upconv1)

    upconv2 = keras.layers.concatenate([upsamp2, conv3])
    upconv2 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(upconv2)
    upconv2 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(upconv2)
    upsamp3 = keras.layers.UpSampling2D((2, 2))(upconv2)

    upconv3 = keras.layers.concatenate([upsamp3, conv2])
    upconv3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(upconv3)
    upconv3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(upconv3)
    upsamp4 = keras.layers.UpSampling2D((2, 2))(upconv3)

    upconv4 = keras.layers.concatenate([upsamp4, conv1])
    upconv4 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(upconv4)
    upconv4 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(upconv4)

    conv_out = keras.layers.Conv2D(2, (1,1), padding="same", activation="softmax")(upconv4)

    model = keras.Model(inputs=input_layer, outputs=conv_out)
    model.compile(optimizer = keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=[dice_similarity])

    return model
    