import keras
from dataset import *
import matplotlib.pyplot as pyplot

# Loads the saved model
Unet_model = keras.models.load_model("UNet_model")

# Evaluates the model trained before
Unet_model.evaluate(X_test, Y_test)

"""
    # This assigns the predicted value of images
    # Then the array is rounded into decimal values
    # The maximum axis value is assigned
"""
out = Unet_model.predict(X_test)
out_r = np.round(out)
out_argmax = np.argmax(out, -1)
gt_test_Y = np.argmax(Y_test, -1)
