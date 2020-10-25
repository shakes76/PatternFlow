import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import layers_model as layers

PATH_ORIGINAL_DATA = "data/image"
PATH_SEG_DATA = "data/mask"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
SEED = 45
BATCH_SIZE = 32
EPOCHS = 50
STEPS_PER_EPOCH_TRAIN = 2076
STEPS_PER_EPOCH_TEST = 518
DATA_GEN_ARGS = dict(
    rescale=1.0/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)
TEST_TRAIN_GEN_ARGS = dict(
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))


class ImageMaskSequence(keras.utils.Sequence):
    def __init__(self, image_gen, mask_gen):
        self.generators = [image_gen, mask_gen]

    def __len__(self):
        return len(self.generators[1])

    def __getitem__(self, item):
        return [generator[item] for generator in self.generators]


if __name__ == "__main__":
    image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS)
    mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS)

    image_train_gen = image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='rgb')

    image_test_gen = image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='validation',
        color_mode='rgb')

    mask_train_gen = mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='grayscale')

    mask_test_gen = mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='validation',
        color_mode='grayscale')

    train_gen = ImageMaskSequence(image_train_gen, mask_train_gen)
    test_gen = ImageMaskSequence(image_test_gen, mask_test_gen)

    model = layers.improvedUNet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    track = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        shuffle=True,
        verbose=2)

    test_loss, test_accuracy = model.evaluate(test_gen)

    print("COMPLETED.")
