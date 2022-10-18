from dataset import *
from modules import *
from keras import backend as K
import matplotlib.pyplot as plt


unet_model = build_unet_model()

def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=dice_coef_loss,
                   metrics=[dice_coef])

unet_model.summary()


unet_model.fit(X_train, Y_train, batch_size=8, epochs=3,
               validation_data=(X_validate, Y_validate))


unet_model.save("UNet_model")


plt.plot(unet_model.history.history['dice_coef'])
plt.plot(unet_model.history.history['val_dice_coef'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


plt.plot(unet_model.history.history['loss'])
plt.plot(unet_model.history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
