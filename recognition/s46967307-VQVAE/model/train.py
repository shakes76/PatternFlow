print("Beginning Training")
from modules import AE
from dataset import load_data
import tensorflow as tf
#print(tf.config.list_physical_devices())

# Clear graph
#tf.keras.backend.clear_session()

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
# Data has lengths:
# Train - 544
# Test - 9664
# Validate - 1120

# Begin model training and validation.
print("Beginning Model Fitting")
model.fit((data["train"]),
        (data["train"]),
        epochs=2,
        shuffle=True,
        batch_size=8)
print("Finished MOdel Fitting")

#predictions = model.predict(data["test"])

import matplotlib.pyplot as plt
n = 5
plt.tight_layout()
fig, axs = plt.subplots(n, 2, figsize=(256,256))
for i in range(n):
    noise = tf.random.uniform(shape=(None,32,32,8))
    noise2 = tf.random.uniform(shape=(None,32,32,8))
    axs[i,0].imshow(tf.reshape(model.decoder.predict(model.vq.predict(noise)), shape=(256,256)))
    axs[i,1].imshow(tf.reshape(model.decoder.predict(model.vq.predict(noise2)), shape=(256,256)))
plt.savefig("out.png", dpi=50)

# Save model.
print("Saving Model")
model.save("model.ckpt")
print("Model Saved to model.ckpt")
