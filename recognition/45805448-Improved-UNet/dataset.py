import tensorflow as tf
from keras.layers import Rescaling, RandomFlip, ZeroPadding2D, RandomCrop

def merge_image_mask_datasets(image_dataset, mask_dataset):
    """
    Merges x and y datasets into one for easier usage.

    Returns:
        merged dataset
    """
    return tf.data.Dataset.zip((image_dataset, mask_dataset))

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

def load_image_dataset_from_directories(directories, image_size=(64, 64)):
    """
    Loads images from several directories and concatenates them into one cached dataset. 

    Returns:
        cached dataset of images
    """
    datasets = []
    for directory in directories:
        datasets.append(load_image_dataset_from_directory(directory, image_size))

    image_dataset = datasets[0]
    for dataset in datasets[1:]:
        image_dataset = image_dataset.concatenate(dataset)

    return image_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def load_image_dataset_from_directory(directory, image_size=(64, 64)):
    """
    Loads a set of images from a directory.

    Returns:
        dataset of images
    """
    return tf.keras.utils.image_dataset_from_directory(directory, labels=None, color_mode='rgb', 
                                    batch_size=3, image_size=image_size, shuffle=True)

def preprocess_dataset(dataset, seed=420):
    """
    Preprocesses an entire dataset.

    Returns:
        the preprocessed dataset
    """
    tf.random.set_seed(seed)
    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

def preprocess(images):
    """
    Preprocess the images by applying normalization, random horizontal & 
    vertical flips, and random cropping.

    Returns:
        similarly augmented images and masks
    """
    # Normalization
    normalize = Rescaling(1./255)(images)

    # Random flipping
    horizontal_flip = RandomFlip(mode='horizontal')(normalize)
    vertical_flip = RandomFlip(mode='vertical')(horizontal_flip)

    # Random cropping
    _, height, width, _ = images.shape
    pad = ZeroPadding2D(padding=4)(vertical_flip)
    crop = RandomCrop(height=height, width=width)(pad)

    return crop
