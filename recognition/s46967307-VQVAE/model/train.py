print("Beginning Training")
from modules import AE
from dataset import load_data
import tensorflow as tf

# Clear graph
tf.keras.backend.clear_session()

# Initialize the Model, Print a summary.
print("Loading Model")
model = AE()
model.compile(optimizer='adam')
print("Finished Loading Model")
print(model.encoder.summary())
print(model.vq.summary())
print(model.decoder.summary())

# Load the Data.
print("Loading data")
data = load_data()
print("Finished Loading Data")

# Begin model training and validation.
print("Beginning Model Fitting")
model.fit((data["train"]),
        (data["train"]),
        epochs=5,
        shuffle=True,
        batch_size=32)
print("Finished MOdel Fitting")

# Save model.
print("Saving Model")
model.save("model.ckpt")
print("Model Saved to model.ckpt")
