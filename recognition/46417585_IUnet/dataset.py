import tensorflow as tf

# fmt: off
test = tf.data.Dataset.list_files("isic_dataset/ISIC-2017_Test_v2_Data/ISIC_*.jpg")
train = tf.data.Dataset.list_files("isic_dataset/ISIC-2017_Training_Data/ISIC_*.jpg")
validation = tf.data.Dataset.list_files("isic_dataset/ISIC-2017_Validation_Data/ISIC_*.jpg")
# fmt: on


def parse_dataset(dataset: tf.data.Dataset):
    from operator import truediv as divide

    return (
        dataset.map(tf.io.read_file)
        .map(tf.image.decode_jpeg)
        .map(lambda img: tf.image.resize(img, (256, 256)))
        .map(lambda img: divide(img, 255))
    )
