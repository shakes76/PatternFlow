import tensorflow_datasets as tfds
import aoi_akoa


def test_no_shared_patients():
    dataset = tfds.load("aoi_akoa")

    splits = {
        key: set(split.map(lambda x: x["patient_id"]).as_numpy_iterator())
        for key, split in dataset.items()
    }

    assert splits["train"].isdisjoint(splits["validation"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["validation"].isdisjoint(splits["test"])


def test_labels_are_balanced():
    dataset = tfds.load("aoi_akoa")

    splits = {
        key: list(split.map(lambda x: x["label"]).as_numpy_iterator())
        for key, split in dataset.items()
    }

    ratio_split = lambda xs: sum(map(lambda x: 1 - x, xs)) / sum(xs)

    ratio_train = ratio_split(splits["train"])
    ratio_validation = ratio_split(splits["validation"])
    ratio_test = ratio_split(splits["test"])

    # test is exactly 50:50
    assert ratio_test == 1
    # train and validation are pretty well balanced
    assert abs(ratio_train - ratio_validation) < 0.02


def test_ratios():
    dataset = tfds.load("aoi_akoa")

    splits = {key: int(split.cardinality()) for key, split in dataset.items()}

    total = sum(splits.values())

    # splits are approx 70:15:15
    assert abs(0.7 - splits["train"] / total) < 0.01
    assert abs(0.15 - splits["validation"] / total) < 0.01
    assert abs(0.15 - splits["test"] / total) < 0.01
