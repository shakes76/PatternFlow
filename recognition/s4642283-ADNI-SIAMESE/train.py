import tensorflow as tf
import os

from dataset import data_loader
from modules import siamese_model

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load Data
x_train, labels_train, x_val, labels_val, x_test, labels_test, X_data, X_data_labels = data_loader()

# Save model checkpoints
checkpoint_path = os.path.join(__location__, "training/cp.ckpt")

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 40
batch_size = 36

# Create siamese model
siamese, embedding_network = siamese_model((240, 256, 1))

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

# Test Model (note train.py will also test, since the report task sheet specifies
# that the train.py does training, validating and testing). Hence, I will treat
# predict.py as a way to make predictions with the model on the data the user wants.
print("Finished!\n")
siamese.evaluate([x_test[0], x_test[1]], labels_test, batch_size=batch_size)

# Plot Model History (Accuracy and Loss for train and validation data)
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

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Save an image classifier (which classifies an image as having Alzheimer's
# or not)
embedding_network.trainable = False
classifier_input = layers.Input((240, 256, 1))
classifier = embedding_network(classifier_input)
output_class = layers.Dense(2, activation="softmax")(classifier)
classifier = keras.Model(inputs=classifier_input, outputs=output_class)
classifier.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

# Test image classifier
classifier.evaluate(X_data, X_data_labels)
classifier.save(os.path.join(__location__, "Classifier_Model"))