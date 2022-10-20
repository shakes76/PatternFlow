import keras
from dataset import *

Unet_model = keras.models.load_model("UNet_model")

Unet_model.evaluate(X_test, Y_test)
