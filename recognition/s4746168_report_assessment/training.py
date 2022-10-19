from dataset import *
from modules import *
from keras import backend as K
import matplotlib.pyplot as plt

# Unet model is being called from the file modules
unet_model = build_unet_model()

"""
    # This function is used to calculate dice coefficient
    # It take the true and predicted value by the model and compares each of them
    # For more details on coefficient refer README.md on github
"""


def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


"""
    # The Unet model is being compiled here
    # Adam optimiser is used as it is the most suitable one for the model
    # Dice coefficient loss and metrics are used for image segmentation
"""
unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=dice_coef_loss,
                   metrics=[dice_coef])

"""
    # Summary of model is printed for checking the layers 
"""
unet_model.summary()

"""
    # Model is being fitted & trained
    # I have used 10 epochs
    # Validation that was loaded earlier in dataset.py is being used for validating the model
"""
unet_model.fit(X_train, Y_train, batch_size=8, epochs=3,
               validation_data=(X_validate, Y_validate))

# Calling `save('UNet_model')` creates a SavedModel folder `UNet_model`.
unet_model.save("UNet_model")

# Plotting the graph for dice coefficient
plt.plot(unet_model.history.history['dice_coef'])
plt.plot(unet_model.history.history['val_dice_coef'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# Plotting the graph for loss
plt.plot(unet_model.history.history['loss'])
plt.plot(unet_model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
