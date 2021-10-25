"""Training the UNet model
"""
import tensorflow.keras.backend as K
from tensorflow.keras import models, optimizers

def dice_coef(y_true, y_pred) -> float:
    # flatten array for faster computation
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersect = K.sum(K.abs(y_true * y_pred))
    total = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))
    return (2. * intersect + 1.) / (total + 1.)


def dice_loss(y_true, y_pred) -> float:
    return 1 - dice_coef(y_true, y_pred)

def scheduler(epoch, lr):
    pass

def compile_model(model: models.Model):
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4, ),
                  loss=dice_loss, metrics=["accuracy", dice_coef])

def fit_model(model: models.Model, x, y, batch_size, epochs):
    model.fit(x, y, batch_size, epochs, callbacks=[])
