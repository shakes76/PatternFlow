import tensorflow as tf
from keras.layers import Rescaling, RandomFlip, ZeroPadding2D, RandomCrop

def merge_image_mask_datasets(image_dataset, mask_dataset):
    """
    Merges x and y datasets into one for easier usage.

    Returns:
        merged dataset
    """
    return tf.data.Dataset.zip(image_dataset, mask_dataset)

def save_dataset(dataset: tf.data.Dataset, filepath):
    """
    Saves a TensorFlow dataset at the filepath.
    """
    dataset.save(filepath)

def load_dataset(filepath):
    """
    Loads an existing TensorFlow dataset.

    Returns:
        a loaded dataset
    """
    return tf.data.Dataset.load(filepath).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def load_image_dataset_from_directories(directories, batch_size=8, image_size=(64, 64), seed=69):
    """
    Loads images from several directories and concatenates them into one cached dataset. 

    Returns:
        cached dataset of images
    """
    datasets = []
    for directory in directories:
        datasets.append(load_image_dataset_from_directory(directory, batch_size, image_size, seed))

    image_dataset = datasets[0]
    for dataset in datasets[1:]:
        image_dataset = image_dataset.concatenate(dataset)

    return image_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def load_image_dataset_from_directory(directory, batch_size=8, image_size=(64, 64), seed=69):
    """
    Loads a set of images from a directory.

    Returns:
        dataset of images
    """
    return tf.keras.utils.image_dataset_from_directory(directory, labels=None, color_mode='grayscale', 
                                    batch_size=batch_size, image_size=image_size, shuffle=True, seed=seed)

def preprocess_dataset(dataset):
    """
    Preprocesses an entire dataset.

    Returns:
        the preprocessed dataset
    """
    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

def preprocess(images, masks):
    """
    Preprocess the images and their respective masks by applying normalization, random horizontal & 
    vertical flips, and random cropping.

    Returns:
        similarly augmented images and masks
    """
    seed = tf.random.uniform(shape=[], minval=0, maxval=69, dtype=tf.int32)
    _, height, width, _ = images.shape

    augmentations = tf.keras.Sequential([
        Rescaling(1./255),
        RandomFlip(mode='horizontal', seed=seed),
        RandomFlip(mode='vertical', seed=seed),
        ZeroPadding2D(padding=(4, 4)),
        RandomCrop(height=height, width=width, seed=seed),
    ])

    return augmentations(images), augmentations(masks)
