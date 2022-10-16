import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from modules import AE
from dataset import load_data
import tensorflow as tf
print("Beginning Training")
#print(tf.config.list_physical_devices())

# Clear graph
#tf.keras.backend.clear_session()

# Initialize the Model, Print a summary.
print("Loading Model")
model = AE()
model.compile(optimizer=tf.keras.optimizers.Adam())
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
training_data = (tf.concat([data["train"], data["validate"]], axis=0))
model.fit(training_data,
        training_data,
        epochs=2,
        batch_size=8)
print("Finished Model Fitting")

predictions = model.predict(data["test"])

# Save model.
print("Saving Model")
model.save("model.ckpt")
print("Model Saved to model.ckpt")
