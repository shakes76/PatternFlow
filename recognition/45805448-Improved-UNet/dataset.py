import tensorflow as tf
from keras.layers import Rescaling, RandomFlip, ZeroPadding2D, RandomCrop

def save_dataset(dataset: tf.data.Dataset, filepath):
    dataset.save(filepath)

def load_dataset(filepath):
    return tf.data.Dataset.load(filepath).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def load_image_dataset_from_directories(directories, batch_size=8, image_size=(64, 64), seed=69):
    datasets = []
    for directory in directories:
        datasets.append(load_image_dataset_from_directory(directory, batch_size, image_size, seed))

    image_dataset = datasets[0]
    for dataset in datasets[1:]:
        image_dataset = image_dataset.concatenate(dataset)

    return image_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def load_image_dataset_from_directory(directory, batch_size=8, image_size=(64, 64), seed=69):
    dataset = tf.keras.utils.image_dataset_from_directory(directory, labels=None, color_mode='grayscale', 
                                    batch_size=batch_size, image_size=image_size, shuffle=True, seed=seed)

    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

def preprocess(images):
    """
    Preprocess the images
    """
    _, height, width, _ = images.shape

    normalize = Rescaling(1./255)(images)

    horizontal_flip = RandomFlip(mode='horizontal')(normalize)
    vertical_flip = RandomFlip(mode='vertical')(horizontal_flip)

    pad = ZeroPadding2D(padding=(4, 4))(vertical_flip)
    crop = RandomCrop(height=height, width=width)(pad)

    return crop
