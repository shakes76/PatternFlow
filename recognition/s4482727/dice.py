from tensorflow.keras.losses import Loss
from tensorflow import math


class Dice(Loss):

    def sorensen_dice(self, y_true, y_pred):
        tp = math.reduce_sum(y_true * y_pred)
        allp = math.reduce_sum(y_true) + math.reduce_sum(y_pred)
        return 2 * tp / allp

    def call(self, y_true, y_pred):
        return 1 - self.sorensen_dice(y_true, y_pred)
