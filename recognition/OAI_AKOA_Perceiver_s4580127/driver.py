import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from Perceiver import Perceiver
from matplotlib import pyplot as plt


"""
Loads the images and returns a tf dataset which can be used to train the perceiver.
The image directory folder must contains two subdirectories "left" and "right" which contains their respective images.
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


"""
Plots the model accuracy over epochs
"""
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # (Hyper) Parameters
    num_classes = 2  # 2 Classes: Left or Right
    num_epochs = 10  # 3 epochs appears to be enough to get over 97% test accuracy - more epochs leads to increased performance
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

    # Use the perceiver to classify OAI AKOA Knee data laterality
    train_ds, val_ds, test_ds = load_data("images", resized_image_size)
    perceiver_classifier = Perceiver(patch_size=patch_size, data_dim=(resized_image_size // patch_size) ** 2,
                                     latent_size=latent_size, projection_size=projection_size, num_heads=num_heads,
                                     transformer_depth=transformer_depth, dense_units=dense_units, dropout_rate=dropout_rate,
                                     depth=depth, classifier_units=classifier_units)
    history = perceiver_classifier.compile_and_fit(train_ds, val_ds, num_epochs)

    # Evaluate the performance on the test data set and plot the training loss
    _, accuracy = perceiver_classifier.evaluate(x=test_ds)
    print("Test accuracy:", accuracy)
    plot_history(history)

