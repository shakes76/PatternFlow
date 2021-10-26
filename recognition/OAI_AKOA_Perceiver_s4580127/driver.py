import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_data():
    # Load images, split into train, val, test and normalise to (0, 1)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_ds = image_dataset_from_directory("images", label_mode="binary", color_mode="grayscale", batch_size=22,
                                            image_size=(128, 128), shuffle=True, validation_split=0.3,
                                            subset="training", seed=2).map(lambda x, y: (normalization_layer(x), y))
    val_ds = image_dataset_from_directory("images", label_mode="binary", color_mode="grayscale", batch_size=22,
                                          image_size=(128, 128), shuffle=True, validation_split=0.3,
                                          subset="validation", seed=2).map(lambda x, y: (normalization_layer(x), y))
    test_ds = val_ds.shard(2, 0)
    val_ds = val_ds.shard(2, 1)

    # Configure caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    print(load_data(1))
