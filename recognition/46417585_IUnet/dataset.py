import tensorflow as tf


def parsed_dataset(pathname):
    "With a glob string containing image filepaths, read and decode images, resize them to a 256x256, and normalise colour"
    from glob import glob
    from operator import truediv as divide

    return (
        tf.data.Dataset.from_tensor_slices(sorted(glob(pathname)))
        .map(tf.io.read_file)
        .map(lambda img: tf.io.decode_png(img, channels=1))
        .map(lambda img: tf.image.resize(img, (256, 256)))
        .map(lambda img: divide(img, 255))  # Normalise greyscale colour -> [0, 1]
    )


def binary_encode(img):
    "For a normalised image, binary encodes each pixel in the image"
    return tf.cast(img == 1.0, tf.uint8)


# fmt: off
x_test = parsed_dataset("isic_dataset/ISIC-2017_Test_v2_Data/ISIC_*.jpg")
y_test = parsed_dataset("isic_dataset/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC_*.png").map(binary_encode)

x_train = parsed_dataset("isic_dataset/ISIC-2017_Training_Data/ISIC_*.jpg")
y_train = parsed_dataset("isic_dataset/ISIC-2017_Training_Part1_GroundTruth/ISIC_*.png").map(binary_encode)

x_validation = parsed_dataset("isic_dataset/ISIC-2017_Validation_Data/ISIC_*.jpg")
y_validation = parsed_dataset("isic_dataset/ISIC-2017_Validation_Part1_GroundTruth/ISIC_*.png").map(binary_encode)
# fmt: on


from itertools import islice as take
from utils import DSC

for x in take(y_validation, 1):
    print(x)
