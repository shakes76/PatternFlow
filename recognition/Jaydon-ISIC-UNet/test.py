from model import UNet
from tensorflow.keras.utils import plot_model

# Generate a test UNet model to check if it compiles correctly
model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()
plot_model(model, show_shapes=True)
