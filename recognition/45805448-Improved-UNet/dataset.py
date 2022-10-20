import tensorflow as tf
from keras.layers import Rescaling, RandomFlip, ZeroPadding2D, RandomCrop, minimum, concatenate

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
    return tf.data.Dataset.load(filepath).prefetch(buffer_size=tf.data.AUTOTUNE)

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

    return image_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def load_image_dataset_from_directory(directory, image_size=(64, 64), batch_size=5, color_mode='rgb'):
    """
    Loads a set of images from a directory.

    Returns:
        dataset of images
    """
    return tf.keras.utils.image_dataset_from_directory(directory, labels=None, color_mode=color_mode, 
                                    batch_size=batch_size, image_size=image_size, shuffle=True, seed=69)

def preprocess_dataset(dataset, seed=420):
    """
    Preprocesses an entire dataset.

    Returns:
        the preprocessed dataset
    """
    tf.random.set_seed(seed)
    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

def preprocess(images, masks):
    """
    Preprocess the images by applying normalization, random horizontal & 
    vertical flips, and random cropping.

    Returns:
        similarly augmented images and masks
    """
    # Normalization
    normalize = Rescaling(1./255)(images)
    bound = Rescaling(1./255)(masks)
    # One-Hot Encoding
    categorical = tf.squeeze(tf.one_hot(tf.cast(tf.round(bound), dtype=tf.int32), depth=2))
    # Concatenate to ensure consistence randomisation
    images_masks = tf.concat([normalize, categorical], -1)
    # Random flipping
    flips = RandomFlip()(images_masks)
    # Random cropping
    _, height, width, _ = images.shape
    pad = ZeroPadding2D(padding=8)(flips)
    crop = RandomCrop(height=height, width=width)(pad)

    # Return to normal shape
    images = crop[:,:,:,0:3]
    masks = crop[:,:,:,3:]

    return images, masks