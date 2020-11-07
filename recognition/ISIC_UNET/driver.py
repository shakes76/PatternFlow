"""
Loads images from the ISIC dataset and displays some example images.
Creates and trains a UNet model to segment these images.
Evaluates the model on the test set and displays some example results.
"""
#%%
### LOCATIONS TO CHANGE ###

# Code folder location
code_root_folder = 'H:\\COMP3710\\PatternFlow\\recognition\\ISIC_UNET'
# Image folder location
image_root_folder = 'C:\\Users\\s4537175\\Downloads\\COMP3710'

#%%
### Imports ###
import tensorflow as tf

#%%
# Add code location to system location to enable local imports
import sys
print(sys.path)
sys.path.append(code_root_folder)
print(sys.path)

#%%
# Local imports
from load_images import get_datasets, view_imgs, view_preds
from model import UNetModel
from improved_model import ImprovedUNetModel
from metrics import *

#%%
### Loading and splitting image set ###

# Proportions of image set to use
total_prop = 1
val_prop = 0.1
test_prop = 0.1

# Create datasets from images, based on given proportions
train_ds, val_ds, test_ds = get_datasets(image_root_folder, 
                                         total_prop, val_prop, test_prop)

# View some training images and their segmentation
view_imgs(train_ds, 3)

#%%
### Creating and training model ###

# Number of filters in first layer of model
filters = 16
# Batch size model will train on
batch_size = 16
# Build model
#model = UNetModel(filters)
model = ImprovedUNetModel(filters)
model.build((batch_size, 256,256,3))
model.summary() 

#%%

adam_opt = tf.keras.optimizers.Adam(learning_rate=5*10**(-5))

model.compile(optimizer=adam_opt,
              loss='binary_crossentropy',
              metrics=[dsc_fore, dsc_back])

#%%
history = model.fit(train_ds.batch(batch_size),
                    validation_data=val_ds.batch(batch_size),
                    epochs=3)

#%%
### Evaluating model and viewing results ###

# Evalute model metrics on test set
model.evaluate(test_ds.batch(batch_size))

#%%
# View some test images, their true segmentations, and their unrounded and
# rounded predicted segmentations
view_preds(model, test_ds, 3)

#%%
k = 10
eek1 = 0
eek2 = 0
for img, true_segs in train_ds.take(k):
        predictions = model.predict(tf.reshape(img, [1, 256, 256, 3]))
        pred_segs = tf.reshape(predictions, [256, 256, 2])
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
