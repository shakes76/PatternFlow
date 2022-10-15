import tensorflow as tf


def parsed_dataset(pathname):
    "With a glob string containing image filepaths, read and decode images, resize them to a 256x256, and normalise colour"
    from glob import glob
    from operator import truediv as divide

    return (
        tf.data.Dataset.from_tensor_slices(sorted(glob(pathname)))
        .map(tf.io.read_file)
        .map(tf.image.decode_jpeg)
        .map(lambda img: tf.image.resize(img, (256, 256)))
        .map(lambda img: divide(img, 255))
    )


x_test, y_test, x_train, y_train, x_validation, y_validation = map(
    parsed_dataset,
    [
        "isic_dataset/ISIC-2017_Test_v2_Data/ISIC_*.jpg",
        "isic_dataset/ISIC-2017_Test_v2_Data/ISIC_*.png",
        "isic_dataset/ISIC-2017_Training_Data/ISIC_*.jpg",
        "isic_dataset/ISIC-2017_Training_Data/ISIC_*.png",
        "isic_dataset/ISIC-2017_Validation_Data/ISIC_*.jpg",
        "isic_dataset/ISIC-2017_Validation_Data/ISIC_*.png",
    ],
)
