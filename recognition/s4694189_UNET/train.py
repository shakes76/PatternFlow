from dataset import *
from module import *
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

# Unet model is being called from the file modules
#model = unet_model()
model = unet_model()

def dice_coef(y_true, y_pred, smooth = 1e-15):
    y_true = tf.convert_to_tensor(y_true, dtype='float32')
    y_pred = tf.convert_to_tensor(y_pred, dtype='float32')
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [dice_coef])
model.summary()

model.fit(X_train, masks_train_images, batch_size=8, epochs=10,
               validation_data=(X_validate, masks_valid_images))


model.save("improved_UNET.h5")

# Plotting our loss charts

# Use the History object we created to get our saved performance results
history_dict = model.history.history

# Extract the loss and validation losses
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

# Get the number of epochs and create an array up to that number using range()
epochs = range(1, len(loss_values) + 1)

# Plot line charts for both Validation and Training Loss
line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


# Plotting the graph for dice coefficient
plt.plot(model.history.history['dice_coef'])
plt.plot(model.history.history['val_dice_coef'])
plt.title('model dice coefficient')
plt.ylabel('dice coefficient')
plt.xlabel('epoch')
plt.show()






