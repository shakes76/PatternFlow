import os
from skimage import io, img_as_ubyte
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import *


def evaluate(data_dir, valid_dir, seg_valid_dir):
    # image data generator parameters
    args = {
        'rotation_range': 0.2,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.05,
        'zoom_range': 0.05,
        'horizontal_flip': True,
        'fill_mode': 'nearest',
        'rescale': 1./255
    }

    # create image data generator for validation data
    valid_datagen = ImageDataGenerator(**args)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir,
        classes=[valid_dir],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='image',
        seed=1)

    # create image data generator for validation data mask
    seg_valid_datagen = ImageDataGenerator(**args)
    seg_valid_generator = seg_valid_datagen.flow_from_directory(
        data_dir,
        classes=[seg_valid_dir],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='image',
        seed=1)

    # create validation generator object
    validf_generator = zip(valid_generator, seg_valid_generator)

    # evaluate model on test data
    model.evaluate(validf_generator, steps=1120)

def predict(data_dir, test_dir, results_dir):
    # image data generator parameters
    args = {
        'rotation_range': 0.2,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.05,
        'zoom_range': 0.05,
        'horizontal_flip': True,
        'fill_mode': 'nearest',
        'rescale': 1./255
    }

    # create image data generator for test data
    test_datagen = ImageDataGenerator(**args)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        classes=[test_dir],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=1,
        save_to_dir=None,
        save_prefix='image',
        seed=1)

    # use model for prediction on test generator
    results = model.predict(test_generator, steps=544, verbose=1)

    # save prediction results
    for i, item in enumerate(results):
        img = item[:, :, 0]
        io.imsave(os.path.join(f'{data_dir}{results_dir}', '%d_predict.png' % i), img_as_ubyte(img))


if __name__ == '__main__':
    data_directory = 'keras_png_slices_data\\'
    
    train_directory = 'keras_png_slices_train'
    seg_train_directory = 'keras_png_slices_seg_train'
    valid_directory = 'keras_png_slices_validate'
    seg_valid_directory = 'keras_png_slices_seg_validate'
    test_directory = 'keras_png_slices_test'
    seg_test_directory = 'keras_png_slices_seg_test'

    results_directory = 'results\\'

    # train model
    model = train_model(data_directory, train_directory, seg_train_directory)

    # evaluate model on validation data
    evaluate(data_directory, valid_directory, seg_valid_directory)

    # predict on test data
    predict(data_directory, test_directory, results_directory)