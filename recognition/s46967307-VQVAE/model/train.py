print("Beginning Training")
from modules import AE
from dataset import load_data
import tensorflow as tf

# Initialize the Model, Print a summary.
model = AE()
model.compile(optimizer='adam')
print("Model Summary:")
print(model.encoder.summary())
print(model.decoder.summary())

# Load the Data.
data = load_data()
print("Finished Loading Data")

# Begin model training and validation.
model.fit((data["train"]),
        (data["train"]),
        epochs=5,
        shuffle=True,
        batch_size=32)

# Save model.
model.save("CKPT")
