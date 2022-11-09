"""
Loads images from the ISIC dataset and displays some example images.
Creates and trains an Improved UNet model to segment these images.
Evaluates the model on the test set and displays some example results.
"""
### LOCATIONS TO CHANGE ###

# Code folder location
code_root_folder = 'H:\\COMP3710\\PatternFlow\\recognition\\ISIC_UNET'
# Image folder location
image_root_folder = 'C:\\Users\\s4537175\\Downloads\\COMP3710'

### Imports ###
import tensorflow as tf

# Add code location to system location to enable local imports
import sys
sys.path.append(code_root_folder)

# Local imports
from load_images import get_datasets, view_imgs, view_preds
from model import UNetModel
from metrics import *

if __name__ == '__main__':
    ### Loading image set into split datasets ###

    # Proportions of image set to use
    total_prop = 1
    val_prop = 0.1
    test_prop = 0.1

    # Create datasets from images, based on given proportions
    train_ds, val_ds, test_ds = get_datasets(image_root_folder, 
                                             total_prop, val_prop, test_prop)

    # View some training images and their segmentation maps
    view_imgs(train_ds, 3)

    ### Creating and training model ###

    # Number of filters in first layer of model
    filters = 12
    # Batch size model will train on
    batch_size = 16
    # The size of the images
    # Must match the image size specifed in load_images.py
    image_size = 256

    # Build model
    model = UNetModel(filters)
    model.build((batch_size,image_size,image_size,3))
    model.summary() 

    # Compile model with adam optimiser
    adam_opt = tf.keras.optimizers.Adam(learning_rate=5*10**(-5))
    model.compile(optimizer=adam_opt,
                  loss='categorical_crossentropy',
                  metrics=[dsc_fore, dsc_back])

    # Train model
    history = model.fit(train_ds.batch(batch_size),
                        validation_data=val_ds.batch(batch_size),
                        epochs=3)

    ### Evaluating model and viewing results ###

    # Evalute model metrics on test set
    model.evaluate(test_ds.batch(batch_size))

    # View some test images, their true segmentations, and their unrounded and
    # rounded predicted segmentations
    view_preds(model, test_ds, 3)
