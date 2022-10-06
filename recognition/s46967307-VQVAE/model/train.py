from modules import AE
from dataset import load_data
import tensorflow as tf

# Initialize the Model, Print a summary.
model = AE()
model.compile(optimizer='adam')
print(model.encoder.summary())
print(model.decoder.summary())

# Load the Data.
data = load_data()

# Begin model training and validation.
model.fit((tf.concat(data["train"], data["test"])), 
        (tf.concat(data["train"], data["test"])),
        epochs=5,
        shuffle=True,
        batch_size=32)

# Save model.
model.save("CKPT")
