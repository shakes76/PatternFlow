import tensorflow as tf
import tensorflow.keras.preprocessing.image as image_preprocessing
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def hflip(image, label):
    """Horizontally flip image and reverse label."""
    assert len(image.shape) == 3
    return tf.image.flip_left_right(image), 1 - label


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
    num_classes: int = 2,
    image_dims: tuple[int, int] = None,
    hflip_concat: bool = True,
    train_batch_size: int = 64,
    eval_batch_size: int = 16,
):
    """Pre-process images for each split.

    Applied to all splits:
    - image normalisation
    - one-hot label encoding
    - crop and resize (without distortion)

    Train & validation:

    If `hflip_concat`, training and validation is duplicated by:
    - horizontally flipping images
    - flipping the label
    - concatenating with original splits

    Args:
        train: Training split.
        validation: Validation split.
        test: Testing / evaluation split.
        num_classes: Number of classes / labels.
        image_dims: Image resize dimensions. If none, images are not resized.
        hflip_concat: If true, for *train* and *validation* examples, horizontally
            flip images and labels and concatenate input. This ensures the
            *train* and *validation* sets have balanced classes and doubles the
            number of examples in these splits.
        train_batch_size: Size of training and validation batches.
        eval_batch_size: Size of test / evaluation batches.

    """

    _one_hot = partial(one_hot, num_classes=num_classes)
    _smart_resize = (
        partial(smart_resize, image_dims=image_dims)
        if image_dims is not None
        else lambda image, label: (image, label)
    )

    # TRAIN & VALIDATION: flip images and labels, concat with originals
    if hflip_concat:
        train, validation = [
            split.concatenate(split.map(hflip, AUTOTUNE))
            for split in [train, validation]
        ]

    return [
        split.map(image_norm, AUTOTUNE)
        .map(_smart_resize, AUTOTUNE)
        .map(_one_hot, AUTOTUNE)
        .cache()
        # arbitrary max buffer size
        .shuffle(min(split.cardinality() // 4, 5120))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
        for split, batch_size in [
            (train, train_batch_size),
            (validation, train_batch_size),
            (test, eval_batch_size),
        ]
    ]
