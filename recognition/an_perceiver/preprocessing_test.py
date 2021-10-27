import tensorflow as tf
import tensorflow_datasets as tfds
import aoi_akoa
import preprocessing
from functools import partial

get_dataset = partial(
    tfds.load,
    name="aoi_akoa",
    split=["train", "validation", "test"],
    with_info=True,
    as_supervised=True,
    shuffle_files=True,
)


def test_hflip_concat():
    (train, val, test), info = get_dataset()
    ptrain, pval, ptest = preprocessing.preprocess(
        train,
        val,
        test,
        hflip_concat=True,
        train_batch_size=1,
        eval_batch_size=1,
    )

    def unique(xs):
        xs_tensor = tf.convert_to_tensor(list(xs))
        xs_unique = tf.unique_with_counts(xs_tensor)
        return list(xs_unique.count.numpy())

    test_labels = unique(test.map(lambda _, x: x))

    ptrain_labels, pval_labels, ptest_labels = [
        unique(split.unbatch().map(lambda _, x: tf.argmax(x)))
        for split in [ptrain, pval, ptest]
    ]

    # hflip doesn't affect test
    assert test.cardinality() == ptest.cardinality()
    assert test_labels == ptest_labels

    # train & validation are duplicated, labels should be balanced
    assert 2 * train.cardinality() == ptrain.cardinality()
    assert 2 * val.cardinality() == pval.cardinality()

    assert ptrain_labels[0] == ptrain_labels[1]
    assert pval_labels[0] == pval_labels[1]


def test_resize():
    resize_dims = (128, 128)
    splits, info = get_dataset()
    splits = preprocessing.preprocess(*splits, image_dims=resize_dims)

    for split in splits:
        index_dims = split.element_spec[0].shape[1:-1]
        assert index_dims == resize_dims


def test_one_hot():
    splits, info = get_dataset()
    splits = preprocessing.preprocess(*splits)

    for split in splits:
        label_dims = split.element_spec[1].shape[1:]
        assert label_dims == info.features["label"].num_classes
