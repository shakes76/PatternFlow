import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    args = {
        'rotation_range': 0.2,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.05,
        'zoom_range': 0.05,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }

    image_datagen = ImageDataGenerator(**args)
    mask_datagen = ImageDataGenerator(**args)

    image_generator = image_datagen.flow_from_directory(
        'keras_png_slices_data\\',
        classes=['keras_png_slices_seg_train'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='image',
        seed=1)

    mask_generator = mask_datagen.flow_from_directory(
        'keras_png_slices_data\\',
        classes=['keras_png_slices_seg_test'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='mask',
        seed=1)
