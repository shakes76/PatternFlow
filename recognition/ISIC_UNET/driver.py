"""
Loads images from the ISIC dataset.
Creates and trains a UNet model to segment these images.

@author: s4537175
"""
import tensorflow as tf

from load_images import get_datasets, view_imgs, view_preds
from model import UNetModel


### VARIABLES TO CHANGE ###

# Image folder location
root_folder = 'C:\\Users\\s4537175\\Downloads\\COMP3710'

# Proportions of image set to use
total_prop = 1
val_prop = 0.1
test_prop = 0.1

#%%

train_ds, val_ds, test_ds = get_datasets(root_folder, total_prop, val_prop, test_prop)

view_imgs(train_ds, 3)

#%%

# Number of filters
d = 16

model = UNetModel(d)


#%%
model.build((None, 512,512,3))
model.summary() 

#%%

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_ds.batch(16),
                    validation_data=val_ds.batch(16),
                    epochs=2)


test_loss, test_acc = model.evaluate(test_ds.batch(1))
print('Test accuracy:', test_acc)