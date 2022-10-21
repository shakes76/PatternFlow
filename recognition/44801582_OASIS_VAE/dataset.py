import os
import numpy as np
from PIL import Image


def load_data(path, img_limit=False, want_var=False):
    dataset = []

    for i, img in enumerate(os.listdir(path)):
        if img_limit and i > img_limit:
            break
        else:
            img = Image.open(f"{path}/{img}")
            data = np.asarray(img)
            dataset.append(data)

    dataset = np.array(dataset)

    if want_var:
        data_variance = np.var(dataset / 255.0)
    else:
        data_variance = None

    dataset = np.expand_dims(dataset, -1)
    dataset = (dataset / 255.0) - 0.5

    return dataset, data_variance


def oasis_dataset(images= False):
    train, variance = load_data("data/keras_png_slices_data/keras_png_slices_train", images, True)
    test, _ = load_data("data/keras_png_slices_data/keras_png_slices_test", images)
    validate, _ = load_data("data/keras_png_slices_data/keras_png_slices_validate", images)

    return train, test, validate, variance
