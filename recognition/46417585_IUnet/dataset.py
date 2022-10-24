import tensorflow as tf

from constants import BATCH_SIZE, IMG_DIM


def parsed_dataset(pathname):
    "With a glob string containing image filepaths, read and decode images, resize them to a 256x256, and normalise colour"
    from operator import truediv as divide

    return (
        tf.data.Dataset.list_files(pathname, shuffle=False)
        .map(tf.io.read_file)
        .map(lambda img: tf.io.decode_jpeg(img, channels=3))
        .map(lambda img: tf.image.resize(img, (IMG_DIM, IMG_DIM)))
        .map(lambda img: divide(img, 255.0))
    )


# fmt: off
x_train = parsed_dataset("isic_dataset/ISIC-2017_Training_Data/ISIC_*.jpg")
y_train = parsed_dataset("isic_dataset/ISIC-2017_Training_Part1_GroundTruth/ISIC_*.png")
train_data = tf.data.Dataset.zip((x_train, y_train)).batch(BATCH_SIZE)

x_validation = parsed_dataset("isic_dataset/ISIC-2017_Validation_Data/ISIC_*.jpg")
y_validation = parsed_dataset("isic_dataset/ISIC-2017_Validation_Part1_GroundTruth/ISIC_*.png")
validation_data = tf.data.Dataset.zip((x_validation, y_validation)).batch(BATCH_SIZE)

x_test = parsed_dataset("isic_dataset/ISIC-2017_Test_v2_Data/ISIC_*.jpg")
y_test = parsed_dataset("isic_dataset/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC_*.png")
test_data = tf.data.Dataset.zip((x_test, y_test)).batch(BATCH_SIZE)
