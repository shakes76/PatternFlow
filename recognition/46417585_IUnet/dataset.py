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


x_test, y_test, x_train, y_train, x_validation, y_validation = map(
    parsed_dataset,
    [
        "isic_dataset/ISIC-2017_Test_v2_Data/ISIC_*.jpg",
        "isic_dataset/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC_*.png",
        "isic_dataset/ISIC-2017_Training_Data/ISIC_*.jpg",
        "isic_dataset/ISIC-2017_Training_Part1_GroundTruth/ISIC_*.png",
        "isic_dataset/ISIC-2017_Validation_Data/ISIC_*.jpg",
        "isic_dataset/ISIC-2017_Validation_Part1_GroundTruth/ISIC_*.png",
    ],
)


def ground_truth_postprocess(dataset: tf.data.Dataset):
    "For a dataset of images, binary encodes each pixel in the image"
    return dataset.map(lambda img: tf.cast(img == 1.0, tf.uint8))


y_test, y_train, y_validation = map(
    ground_truth_postprocess, [y_test, y_train, y_validation]
)
