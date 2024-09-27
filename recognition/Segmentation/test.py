'''
OASIS brain segmentation test module.
'''
import tensorflow as tf
import glob
import pathlib
import random
import os
import numpy as np
from tensorflow.keras.backend import sum, mean, flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
from IPython.display import clear_output
from models import improved_unet

# For allow gpu growth. Program gets more GPU memory as it needs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    except Exception as error:
        print(e)

def _get_unique_values():
    '''
    Return a sorted list of unique classes in segmented image
    '''
    # load the random segmented image for getting classes
    img = load_img(random.choice(train_masks), color_mode='grayscale')
    # convert to numpy array
    img_array = img_to_array(img)

    # convert to 1D array for conveniently finding uniques (classes)
    result = img_array.flatten()
    return np.unique(result)


def _image_generator_batch(img_files, mask_files, batch_size=32):
    '''
    A processed image generator (batch mode). 
    Return list train image tensors and list mask (one hot encoded) image tensors
    '''
    count = 0
    classes = _get_unique_values()
    while (True):
        img = np.zeros((batch_size, 256, 256, 1)).astype('float') # Grayscale images for training
        mask = np.zeros((batch_size, 256, 256, 4)).astype('float') # mask for validation

        for i in range(count, count+batch_size): #initially from 0 to 32, count = 0. 
            train_img = tf.io.read_file(img_files[i])
            train_img = tf.image.decode_png(train_img, channels=1)
            train_img = tf.image.resize(train_img, (256, 256))
            train_img = tf.cast(train_img, tf.float32) / 255.0

            img[i-count] = train_img #add to array - img[0], img[1], and so on.
                                                   

            train_mask = load_img(mask_files[i], color_mode='grayscale')
            # convert to numpy array
            train_mask = img_to_array(train_mask)
            for index, unique_value in enumerate(classes):
                train_mask[train_mask == unique_value] = index
            train_mask = train_mask.reshape(256, 256)
            train_mask = tf.one_hot(train_mask, classes.size, 1, 0, -1, tf.float32)

            mask[i-count] = train_mask

        count+=batch_size
        if(count+batch_size>=len(img_files)):
            count=0
        
        yield img, mask

def _image_generator(img_files, mask_files):
    '''
    A processed image generator. Return train image tensor and mask (one hot encoded) image tensor
    '''
    count = 0
    classes = _get_unique_values()
    while (True):
        img = np.zeros((256, 256, 1)).astype('float') # Grayscale images for training
        mask = np.zeros((256, 256, 4)).astype('float') # mask for validation

        train_img = tf.io.read_file(img_files[count])
        train_img = tf.image.decode_png(train_img, channels=1)
        train_img = tf.image.resize(train_img, (256, 256))
        train_img = tf.cast(train_img, tf.float32) / 255.0

        img = train_img #add to array - img[0], img[1], and so on.
                                                   

        train_mask = load_img(mask_files[count], color_mode='grayscale')
        # convert to numpy array
        train_mask = img_to_array(train_mask)
        for index, unique_value in enumerate(classes):
            train_mask[train_mask == unique_value] = index
        train_mask = train_mask.reshape(256, 256)
        train_mask = tf.one_hot(train_mask, classes.size, 1, 0, -1, tf.float32)

        mask = train_mask    

        count += 1
        if(count + 1 >= len(img_files)):
            count=0
        
        yield img, mask

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask', 'Difference Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    '''
    Turn pixel class prediction to pixel value.
    '''
    pred_mask = tf.argmax(pred_mask, axis=-1).numpy()
    for index, value in enumerate(classes):
        pred_mask[pred_mask[:,:,:] == index] = value
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        count = 0
        for image, mask in dataset:
            pred_mask = create_mask(model.predict(image))
            display([image[0], mask[0], pred_mask, np.absolute(mask[0] - pred_mask)]) #pred_mask
            count += 1
            if count > num:
                break
    else:
        norm_sample = sample_image / 255
        norm_sample = norm_sample[tf.newaxis, ...]
        pred_mask = create_mask(model.predict(norm_sample))
        display([sample_image, sample_mask, pred_mask, np.absolute(sample_mask - pred_mask)])

def dice_coef_multilabel(y_true, y_pred, numClass = 4):
    dice=0
    smooth = 1
    # Image in shape (batch, height, width, channels)
    for index in range(numClass):
        y_true_f = flatten(y_true[:,:,:,index])
        y_pred_f = flatten(y_pred[:,:,:,index])
        intersection = sum(y_true_f * y_pred_f)
        union = sum(y_true_f) + sum(y_pred_f)
        dice += (2. * intersection + smooth)/(union + smooth)
    return dice/numClass # taking average of dice coefficient across all class

def dice_loss(y_true, y_pred, num_class = 4):
    return 1 - dice_coef_multilabel(y_true, y_pred, num_class)



BATCH_SIZE = 1

# Dataset directory in same directory as file. OASIS images is in datasets/OASIS
folder_path = pathlib.Path(__file__).parent.absolute()
dataset_path = folder_path / 'datasets/OASIS'
model_path = folder_path / 'Model'
# Load data
# Each will contains all image directories
train_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_train/*.png")))
train_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_train/*.png")))

test_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_test/*.png")))
test_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_test/*.png")))

val_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_validate/*.png")))
val_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_validate/*.png")))

sample_image = img_to_array(load_img(test_images[0], color_mode='grayscale'))
sample_mask = img_to_array(load_img(test_masks[0], color_mode='grayscale'))
classes = _get_unique_values()
num_class = len(classes)

# Create image generator for loading processed image to model
train_img_generator = _image_generator_batch(train_images, train_masks, BATCH_SIZE)
test_img_generator = _image_generator_batch(test_images, test_masks, BATCH_SIZE)
val_img_generator = _image_generator_batch(val_images, val_masks, BATCH_SIZE)

EPOCHS = 1
STEPS_PER_EPOCH = len(train_images) // BATCH_SIZE # // = integer division.
STEPS_PER_EPOCH_TEST = len(test_images) // BATCH_SIZE # // = integer division.

VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(val_images) // BATCH_SIZE // VAL_SUBSPLITS
TRAINING = 0

dependencies = {
    'LeakyReLU': LeakyReLU(0.01), 
    'dice_loss': dice_loss, 
    'dice_coef_multilabel': dice_coef_multilabel
}

try:
    model = load_model(str(model_path), custom_objects=dependencies)
except OSError:
    TRAINING = 1
except IOError:
    TRAINING = 1
except ImportError:
    TRAINING = 1

if TRAINING:
    model = improved_unet(width=256, height=256, channels=1, output_classes=4, batch_size=BATCH_SIZE)

    model.compile(
        optimizer=Adam(learning_rate=0.0005), 
        loss=dice_loss, 
        metrics=[dice_coef_multilabel])

    checkpoint = ModelCheckpoint(filepath='model_OASIS_IUNET.h5')
    model_history = model.fit(train_img_generator, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=val_img_generator,
                          callbacks=[checkpoint])
    # For saving a final model after training
    model.save(str(pathlib.Path('Model')))
    # Plot loss functions
    epochs = range(EPOCHS)
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

model.summary()
test_history = model.evaluate(test_img_generator, steps=STEPS_PER_EPOCH_TEST)
show_predictions(test_img_generator, 1)