"""
Loads images from the ISIC dataset.
Creates and trains a UNet model to segment these images.

@author: s4537175
"""


#%%
### VARIABLES TO CHANGE ###

# Code folder location
code_root_folder = 'H:\\COMP3710\\PatternFlow\\recognition\\ISIC_UNET'
# Image folder location
image_root_folder = 'C:\\Users\\s4537175\\Downloads\\COMP3710'



#%%
import tensorflow as tf

import sys
print(sys.path)
sys.path.append(code_root_folder)
print(sys.path)

#%%

from load_images import get_datasets, view_imgs, view_preds
from model import UNetModel, dsc, dsc_loss, avg_dsc, avg_dsc_loss


### VARIABLES TO CHANGE ###

# Image folder location
image_root_folder = 'C:\\Users\\s4537175\\Downloads\\COMP3710'

# Proportions of image set to use
total_prop = 0.5
val_prop = 0.1
test_prop = 0.1


#%%

train_ds, val_ds, test_ds = get_datasets(image_root_folder, 
                                         total_prop, val_prop, test_prop)

view_imgs(train_ds, 3)

#%%

# Number of filters
d = 4

model = UNetModel(d)


#%%
model.build((None, 512,512,3))
model.summary() 

#%%

adam_opt = tf.keras.optimizers.Adam(learning_rate= 1*10**(-8))

model.compile(optimizer=adam_opt,
              loss=avg_dsc_loss,
              metrics=['accuracy', avg_dsc])

#%%

history = model.fit(train_ds.batch(32),
                    validation_data=val_ds.batch(32),
                    epochs=2)

#%%

test_loss, test_acc = model.evaluate(test_ds.batch(1))
print('Test accuracy:', test_acc)

#%%

import matplotlib.pyplot as plt
view_preds(model, test_ds, 3)


#%%
for img, true_segs in train_ds.take(1):
        predictions = model.predict(tf.reshape(img, [1, 512, 512, 3]))
        pred_segs = tf.reshape(predictions, [512, 512, 1])
        
        print(avg_dsc(true_segs, pred_segs))