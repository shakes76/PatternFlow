import tensorflow as tf
import tensorflow.keras.preprocessing.image as image_preprocessing
from functools import partial

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


@tf.function
def one_hot(image, label, num_classes=2):
    """One-hot label encoding."""
    return image, tf.one_hot(label, depth=num_classes)


@tf.function
def smart_resize(image, label, image_dims):
    """Resize image to image_dims without distortion."""
    resized = image_preprocessing.smart_resize(image, size=image_dims)
    return resized, label


def preprocess(
    train,
    validation,
    test,
    batch_size: int = 64,
    num_classes: int = 2,
    image_dims: tuple[int, int] = None,
):
    """Preprocess images for each split.

    Applied to all splits:
    - image normalisation
    - one-hot label encoding
    - crop and resize

    If num_classes = 2, training and validation is duplicated by:
    - horizontally flipping images
    - flipping the label
    - concatenating with input
    """

    identity = lambda image, label: (image, label)
    _hflip_concat = hflip_concat if num_classes == 2 else identity
    _one_hot = partial(one_hot, num_classes=num_classes)
    _smart_resize = (
        partial(smart_resize, image_dims=image_dims)
        if image_dims is not None
        else identity
    )

    train, validation = (
        split.map(image_norm, AUTOTUNE)
        .map(_smart_resize, AUTOTUNE)
        .map(_one_hot, AUTOTUNE)
        .map(_hflip_concat, AUTOTUNE)
        .unbatch()
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
        for split in [test, validation]
    )

    test = (
        test.map(image_norm, AUTOTUNE)
        .map(_smart_resize, AUTOTUNE)
        .map(_one_hot, AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train, validation, test
