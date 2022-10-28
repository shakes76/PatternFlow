#!/usr/bin/env python
# coding: utf-8

# In[16]:


from dataset import *
from module import *
from keras import backend as K
import matplotlib.pyplot as plt

# Unet model is being called from the file modules
#model = unet_model()


# In[17]:


model = unet_model()


# In[15]:


smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [dice_coef])
model.summary()


# In[ ]:


model.fit(X_train, masks_train_images, batch_size=8, epochs=10,
               validation_data=(X_validate, masks_valid_images))


# In[ ]:


model.save("improved_UNET.h5")


# In[ ]:


# Plotting our loss charts
import matplotlib.pyplot as plt

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


# In[ ]:


# Plotting the graph for dice coefficient
plt.plot(model.history.history['dice_coef'])
plt.plot(model.history.history['val_dice_coef'])
plt.title('model dice coefficient')
plt.ylabel('dice coefficient')
plt.xlabel('epoch')
plt.show()


# In[ ]:




