import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


import dataset
import modules

modules.model.compile(tf.keras.optimizers.Adam(learning_rate= 0.00003),loss=[modules.dice_loss],metrics=[modules.dice_similarity, 'accuracy'])

history = modules.model.fit(dataset.train_generator , steps_per_epoch=dataset.train_steps ,epochs=15,
                              validation_data=dataset.val_generator,validation_steps=dataset.val_steps, verbose=1)

#Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#dice similarity
plt.plot(history.history['dice_similarity'])
plt.plot(history.history['val_dice_similarity'])
plt.title('model dice_similarity')
plt.ylabel('dice_similarity')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

loss, dice_similarity, acc = model.evaluate(test_generator,batch_size=b_size)

print('Test loss:', loss)
print('Test dice_similarity:', dice_similarity)
print('Test accuracy:', acc)