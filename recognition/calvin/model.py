import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


# function to compute dice similarity coefficient metric
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = 0.12 + K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    return dice


# train unet model
def train_model(data_dir, train_dir, mask_dir):
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

    # create image data generator for training data
    image_datagen = ImageDataGenerator(**args)
    image_generator = image_datagen.flow_from_directory(
        data_dir,
        classes=[train_dir],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='image',
        seed=1)
    
    # create image data generator for training labels
    mask_datagen = ImageDataGenerator(**args)
    mask_generator = mask_datagen.flow_from_directory(
        data_dir,
        classes=[mask_dir],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir=None,
        save_prefix='mask',
        seed=1)

    # create train generator object
    train_generator = zip(image_generator, mask_generator)

    # define unet layers
    inputs = Input((256, 256, 1))

    conv1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(rate=0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(rate=0.5)(conv5)

    ups6 = Conv2D(filters=512, kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    conv6 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(concatenate([drop4, ups6], axis=3))
    conv6 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)

    ups7 = Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    conv7 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(concatenate([conv3, ups7], axis=3))
    conv7 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)

    ups8 = Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    conv8 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(concatenate([conv2, ups8], axis=3))
    conv8 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)

    ups9 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    conv9 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(concatenate([conv1, ups9], axis=3))
    conv9 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(filters=2, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(filters=1, kernel_size=1, padding='valid', activation='sigmoid', kernel_initializer='glorot_uniform')(conv9)

    # train unet model
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[dice_coef])
    model.fit(train_generator, steps_per_epoch=1, epochs=1)

    return model
