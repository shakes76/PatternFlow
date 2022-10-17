"""
"train.py" containing the source code for training, validating, testing and saving my model. The model is
imported from "modules.py" and the data loader is imported from "dataset.py".
"""

# All needed library for training and testing the GCN model.
from sklearn.metrics import classification_report
from dataset import *
from modules import *
import tensorflow as tf
from matplotlib import pyplot

"""
Load the Facebook Large Page-Page Network dataset and preprocess it.
"""
(adjacency_matrix, features_matrix, targets, train_target, validation_target, test_target, train_data_mask,
 validation_data_mask, test_data_mask) = load_facebook_page_data('./facebook.npz')

"""
Build the GCN model
"""
model = GCN(features_matrix)
# multi classes classification so making use of categorical_crossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy', weighted_metrics=['acc'])
model.summary()

"""
Training the GCN model with 100 epochs
"""
history = model.fit([features_matrix, adjacency_matrix], targets, epochs=100, sample_weight=train_data_mask,
                    validation_data=([features_matrix, adjacency_matrix], targets, validation_data_mask),
                    batch_size=features_matrix.shape[0], shuffle=False)

"""
Test the Trained GCN model with test data set
"""
# Prediction of the whole graph
test_result = model.predict([features_matrix, adjacency_matrix], batch_size=features_matrix.shape[0])
print("Test Result with test data set: ")
print(classification_report(np.argmax(test_target, axis=1), np.argmax(test_result[test_data_mask], axis=1)))

"""
Plot the losses and accuracy of the training and validation
"""
# Plot the accuracy of training and validation
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('GCN model training and validation accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['Training', 'Validation'])
pyplot.show()
# Plot the loss of training and validation
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('GCN model training and validation loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['Training', 'Validation'])
pyplot.show()

