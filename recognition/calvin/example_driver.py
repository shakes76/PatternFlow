import os
from skimage import io, img_as_ubyte
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import *


if __name__ == '__main__':
    data_dir = 'keras_png_slices_data\\'
    train_dir = 'keras_png_slices_train'
    mask_dir = 'keras_png_slices_seg_train'

    # train model
    model = train_model(data_dir, train_dir, mask_dir)

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

    # create image data generator for test data (for prediction)
    test_datagen = ImageDataGenerator(**args)
    test_generator = test_datagen.flow_from_directory(
        'keras_png_slices_data\\',
        classes=['keras_png_slices_test'],
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
        io.imsave(os.path.join('keras_png_slices_data\\results\\', "%d_predict.png" % i), img_as_ubyte(img))

    # create image data generator for predicted results (for evaluation)
    test_datagen = ImageDataGenerator(**args)
    test_generator = test_datagen.flow_from_directory(
        'keras_png_slices_data\\',
        classes=['keras_png_slices_test'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='image',
        seed=1)

    # create image data generator for actual results (for evaluation) 
    seg_test_datagen = ImageDataGenerator(**args)
    seg_test_generator = seg_test_datagen.flow_from_directory(
        'keras_png_slices_data\\',
        classes=['keras_png_slices_seg_test'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='image',
        seed=1)

    # create test generator object
    testf_generator = zip(test_generator, seg_test_generator)

    # evaluate model on test data
    model.evaluate(testf_generator, steps=544)
