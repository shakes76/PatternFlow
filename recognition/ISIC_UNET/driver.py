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
from model import *


### VARIABLES TO CHANGE ###


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
d = 8

model = UNetModel(d)


#%%
batch_size = 16
model.build((batch_size, 512,512,3))
model.summary() 

#%%

adam_opt = tf.keras.optimizers.Adam(learning_rate= 1*10**(-5))

model.compile(optimizer=adam_opt,
              loss=avg_dsc_loss,
              metrics=[dsc_fore, dsc_back, 'accuracy'])

#%%
history = model.fit(train_ds.batch(batch_size),
                    validation_data=val_ds.batch(batch_size),
                    epochs=3)

#%%

test_loss, test_acc, b, c = model.evaluate(test_ds.batch(batch_size))
print('Test accuracy:', test_acc)

#%%
view_preds(model, test_ds, 3)

#%%
model.evaluate(train_ds.batch(batch_size))


#%%
k = 10
eek1 = 0
eek2 = 0
for img, true_segs in train_ds.take(k):
        predictions = model.predict(tf.reshape(img, [1, 512, 512, 3]))
        pred_segs = tf.reshape(predictions, [512, 512, 2])
        #print(true_segs[200:210,200:220,1])
        #print(pred_segs[200:210,200:220,1])
        #print(pred_segs[200:210,200:220,0])
        
        #print(avg_dsc(true_segs, pred_segs))
        eek1 = eek1 + avg_dsc(true_segs, pred_segs)
        eek2 = eek2 + dsc_fore(true_segs, pred_segs)
        #print(avg_dsc_loss(true_segs, pred_segs))
        #print(dsc_fore(true_segs, pred_segs))
        #print(dsc_back(true_segs, pred_segs))
print(eek1/k)
print(eek2/k)