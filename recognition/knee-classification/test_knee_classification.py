import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Constants
IMAGE_DIR = './data'
BATCH_SIZE = 32
IMG_SIZE = (260, 228)

"""
Dataset creation function

For this function dataset need to be in a following folder structure
|_ data
   |_ left
      |_ left_image_1
      |_ left_image_2
      |_ ...
   |_ right
      |_ right_image_1
      |_ right_image_2
      |_ ...
"""


def create_dataset(image_dir, batch_size, img_size):
    # Training dataset
    training_dataset = image_dataset_from_directory(image_dir, shuffle=True, validation_split=0.2,
                                                    subset="training", seed=123, color_mode="rgb",
                                                    batch_size=batch_size, image_size=img_size)
    # Validation dataset
    validation_dataset = image_dataset_from_directory(image_dir, shuffle=True, validation_split=0.2,
                                                      subset="validation", seed=123, color_mode="rgb",
                                                      batch_size=batch_size, image_size=img_size)

    # Test dataset
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    # buffered perfecting to load images from disk
    auto_tune = tf.data.experimental.AUTOTUNE
    training_dataset = training_dataset.prefetch(buffer_size=auto_tune)
    validation_dataset = validation_dataset.prefetch(buffer_size=auto_tune)
    test_dataset = test_dataset.prefetch(buffer_size=auto_tune)

    return training_dataset, validation_dataset, test_dataset


if __name__ == "__main__":

    # generate dataset
    training_set, validation_set, test_set = create_dataset(IMAGE_DIR, BATCH_SIZE, IMG_SIZE)

    print('Number of Train batches: %d' % tf.data.experimental.cardinality(training_set))
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_set))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_set))
