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
import matplotlib.pyplot as plt
from IPython.display import clear_output
from models import improved_unet

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


def _image_generator(img_files, mask_files, batch_size=32):
    '''
    A processed image generator. Return train image tensor and mask (one hot encoded) image tensor
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
            # print "randomizing again"
        
        yield img, mask

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(array_to_img(display_list[i]))
        plt.axis('off')

# def create_mask(pred_mask):
#     pred_mask = tf.argmax(pred_mask, axis=-1)
#     pred_mask = pred_mask[..., tf.newaxis]
#     return pred_mask[0]

# def show_predictions(dataset=None, num=1):
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = model.predict(image)
#             display([image[0], mask[0], create_mask(pred_mask)])
#     else:
#         display([sample_image, sample_mask,
#             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

def dice_coef(y_true, y_pred, smooth=1):
    # y_true_f = flatten(y_true)
    # y_pred_f = flatten(y_pred)
    # intersection = sum(y_true_f * y_pred_f)
    # union = sum(y_true_f) + sum(y_pred_f)

    intersection = sum(y_true * y_pred)
    union = sum(y_true) + sum(y_pred)
    
    dice = (2. * intersection + smooth)/(union + smooth)
    # print(dice)
    return dice

def dice_coef_multilabel(y_true, y_pred, numClass = 4):
    dice=0
    # Image in shape (batch, height, width, channels)
    for index in range(numClass):
        # print("Layer {one}".format(one=index))
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numClass # taking average

def dice_loss(y_true, y_pred):
    return 1 - dice_coef_multilabel(y_true, y_pred, 4)



BATCH_SIZE = 1

# Dataset directory in same directory as file. OASIS images is in datasets/OASIS
# dataset_path = pathlib.Path(__file__).parent.absolute() / 'datasets/OASIS'
dataset_path = pathlib.Path("/home/long/projects/COMP3710/Assignment3/PatternFlow/recognition/Segmentation/datasets/OASIS")
# file_location = pathlib.Path(__file__).parent.absolute()
# Load data
# Each will contains all image directories
train_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_train/*.png")))
train_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_train/*.png")))

train_images = train_images[:10]
train_masks = train_masks[:10]

test_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_test/*.png")))
test_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_test/*.png")))

val_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_validate/*.png")))
val_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_validate/*.png")))

sample_image = img_to_array(load_img(test_images[0], color_mode='grayscale'))
sample_mask = img_to_array(load_img(test_masks[0], color_mode='grayscale'))
num_class = len(_get_unique_values())

# Create image generator for loading processed image to model
train_img_generator = _image_generator(train_images, train_masks, BATCH_SIZE)
test_img_generator = _image_generator(test_images, test_masks, BATCH_SIZE)
val_img_generator = _image_generator(val_images, val_masks, BATCH_SIZE)

model = improved_unet(width=256, height=256, channels=1, output_classes=4, batch_size=BATCH_SIZE)

model.summary()
model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss=dice_loss, 
    metrics=[dice_coef_multilabel, 'accuracy'])

checkpoint = ModelCheckpoint(filepath='model_OASIS_IUNET.h5')
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2, mode='min', baseline=0.1)

EPOCHS = 3
STEPS_PER_EPOCH = len(train_images) // BATCH_SIZE # // = integer division.
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(val_images) // BATCH_SIZE // VAL_SUBSPLITS
model_history = model.fit_generator(train_img_generator, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=val_img_generator,
                          callbacks=[checkpoint])

# For saving a final model
model.save('Model.h5')
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

print(loss)
print(val_loss)

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()