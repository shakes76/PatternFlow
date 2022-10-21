import tensorflow as tf
import os

from dataset import data_loader
from modules import siamese_model

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load Data
x_train, labels_train, x_val, labels_val, x_test, labels_test = data_loader()

# Save model checkpoints
checkpoint_path = os.path.join(__location__, "training/cp.ckpt")

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 20
batch_size = 32

# Create siamese model
siamese = siamese_model((240, 256, 1))

# Compile and show model
siamese.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
siamese.summary()

# Train model
history = siamese.fit(
    [x_train[0], x_train[1]],
    labels_train,
    validation_data=([x_val[0], x_val[1]], labels_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[cp_callback]
)

# Save model once training completes
siamese.save(os.path.join(__location__, "SiameseModel"))

# Test Model
print("Finished!\n")
siamese.evaluate([x_test[0], x_test[1]], labels_test, batch_size=batch_size)

# Plot Model History (Acc and Loss)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()