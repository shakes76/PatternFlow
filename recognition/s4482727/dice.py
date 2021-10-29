from tensorflow.keras.losses import Loss
from tensorflow import math


def sorensen_dice(y_true, y_pred):
    tp = math.reduce_sum(y_true * y_pred)
    allp = math.reduce_sum(y_true) + math.reduce_sum(y_pred)
    return 2 * tp / allp


class Dice(Loss):

    def call(self, y_true, y_pred):
        return 1 - sorensen_dice(y_true, y_pred)
