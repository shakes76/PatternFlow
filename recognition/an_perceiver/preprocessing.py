import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def hflip_concat(image, label):
    """Horizontal flip images and labels, concat with input."""
    assert len(image.shape) == 3

    flipped_image = tf.image.flip_left_right(image)
    flipped_label = 1 - label

    return tf.stack([image, flipped_image]), tf.stack([label, flipped_label])


@tf.function
def image_norm(image, label):
    """Normalise images 0-mean, unit variance."""
    assert len(image.shape) == 3

    return tf.image.per_image_standardization(image), label


def preprocess(train, validation, test):
    """Preprocess images for each split.

    Applies image normalisation to all splits.
    Duplicates all train and validation samples by:
    - horizontally flipping images
    - flipping the label
    - concatenating with input
    """
    train, validation = (
        split.map(image_norm, num_parallel_calls=AUTOTUNE)
        .map(hflip_concat, num_parallel_calls=AUTOTUNE)
        .unbatch()
        .prefetch(AUTOTUNE)
        for split in [test, validation]
    )

    test = test.map(image_norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train, validation, test
