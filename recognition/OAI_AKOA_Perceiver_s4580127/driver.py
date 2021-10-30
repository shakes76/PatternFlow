import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from Perceiver import Perceiver


"""
Loads the images and returns a tf dataset which can be used to train the perceiver.
Requires the iamge directory to be on the same level as the driver and Perceiver files and this folder must contains 
two subdirectories "left" and "right" which contains their respective images.
"""
def load_data(image_directory, image_resize):
    # Load images, split into train, val, test and normalise to (0, 1)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_ds = image_dataset_from_directory(image_directory, label_mode="binary", color_mode="grayscale", batch_size=22,
                                            image_size=(image_resize, image_resize), shuffle=True, validation_split=0.3,
                                            subset="training", seed=2).map(lambda x, y: (normalization_layer(x), y))
    val_ds = image_dataset_from_directory(image_directory, label_mode="binary", color_mode="grayscale", batch_size=22,
                                          image_size=(image_resize, image_resize), shuffle=True, validation_split=0.3,
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
    # Hyper parameters
    num_classes = 2  # Left or Right
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 10
    dropout_rate = 0.2  # Dropout rate for dense layers
    resized_image_size = 128  # We'll resize input images to this size.
    patch_size = 2  # Size of the patches to be extracted from the images.
    num_patches = (resized_image_size // patch_size) ** 2  # Number of patches we extract from each image (size of data tensor)
    latent_size = 256  # Size of each latent data vector
    projection_size = 256  # Size of each image-vector in the data and latent tensors
    num_heads = 8  # Number of Transformer heads.
    dense_units = [projection_size, projection_size]  # Size of the Transformer network.
    transformer_depth = 4  # Depth of each transformer module
    depth = 2  # How many data iterations our model performs - each depth has a cross-attention and transformer module.
    classifier_units = [projection_size, num_classes]  # Size of the dense network of the final classifier.
    
    perceiver_classifier = Perceiver(
        patch_size=patch_size,
        data_dim=(resized_image_size // patch_size) ** 2,
        latent_size=latent_size,
        projection_size=projection_size,
        num_heads=num_heads,
        transformer_depth=transformer_depth,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        depth=depth,
        classifier_units=classifier_units,
    )

    train_ds, val_ds, test_ds = load_data("images", resized_image_size)

    # Create LAMB optimizer with weight decay.
    # optimizer = tfa.optimizers.LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model.
    perceiver_classifier.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    # Create a learning rate scheduler callback.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)

    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    # Fit the model.
    history = perceiver_classifier.fit(x=train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=[early_stopping, reduce_lr])

    _, accuracy = perceiver_classifier.evaluate(train_ds, test_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

