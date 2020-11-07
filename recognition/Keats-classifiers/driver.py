import argparse
import tensorflow as tf
import os
from models import KneeClassifier

def get_label(file_path):
    """ Get the data label corresponding to the file path 

    Parameters:
        file_path (str): The file (image) to look at
    
    Returns:
        A tensor with datatype float32 which is 1.0 if the image
        is of the right knee and 0.0 otherwise.
    """
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # Extract LEFT/RIGHT data from filename
    parts = tf.strings.split(parts[-1], '.')
    parts = tf.strings.split(parts[0], '_')
    pattern = tf.constant("right")
    
    is_right = pattern == tf.strings.lower(parts[-1])
    # convert from bool to float
    return tf.cast(is_right, dtype='float32')

def decode_img(img):
    """ Attempts to get a 3D tensor corresponding to the image data

    Parameters:
        img (0-D string): The encoded image bytes

    Returns:
        3D grayscale image tensor with values normalised between 0.0 and 1.0
    """
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, dtype='float32')
    # cast and scale the tensor
    img = tf.math.truediv(img, tf.constant(255.0))
    
    return img #tf.image.resize(img, [HEIGHT, WIDTH])

def process_path(file_path):
    """ Map function to be applied to dataset, 
    returns image and label for a filepath 
    
    Parameters:
        file_path (str): The filepath to generate data for

    Returns:
        (tuple<tf.Tensor, tf.Tensor>) A tuple of tensors for the image and label
    """
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def configure_for_performance(dataset, batch_size=32):
    """ Process the dataset to make it efficient to learn with. For proper learning
    apparently its important to have quick access to batches, shuffle your data and have it 
    cached. I just do what's recommended on the tf site.
    """
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
        
# Data processing
def load_knee_data(path):
    dataset = tf.data.Dataset.list_files(f"{path}/*.png")
    
    # Split the data into training and validation sets
    data_count = len(os.listdir(path))
    VALIDATION_SPLIT = 0.2
    VALIDATION_SIZE = int(VALIDATION_SPLIT * data_count)

    training_data = dataset.skip(VALIDATION_SIZE)
    testing_data = dataset.take(VALIDATION_SIZE)

    # See the size of our sets
    print("Training set cardinality:", tf.data.experimental.cardinality(training_data).numpy())
    print("Validation set cardinality:", tf.data.experimental.cardinality(testing_data).numpy())
    
    # map the filenames to image/label pairs for learning
    training_data = training_data.map(process_path, 
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    testing_data = testing_data.map(process_path, 
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # configure the datasets to make them efficient to learn with
    training_data = configure_for_performance(training_data)
    testing_data = configure_for_performance(testing_data)
    
    print("Training set cardinality:", tf.data.experimental.cardinality(training_data).numpy())
    print("Validation set cardinality:", tf.data.experimental.cardinality(testing_data).numpy())

    return training_data, testing_data

def main():
    # Commandline argument parsing
    parser = argparse.ArgumentParser(description='Create a ConvNet for determining OAI AKOA Knee laterality')
    parser.add_argument('data_directory', help='The directory containing the extracted OAI AKOA data')
    parser.add_argument('--relearn', help='If this flag is specified, relearn the weights on the network', action='store_true')
    args = parser.parse_args()

    # load and preprocess the training data
    training_data, testing_data = load_knee_data(args.data_directory)

    # Create and compile the classifier. Will train from scratch if specified
    classifier = KneeClassifier(args.relearn, training_data, testing_data)
    classifier.show_results(3, 3)

    loss, accuracy = classifier.get_model().evaluate(testing_data, verbose=2)
    print("Model accuracy on validation data: {:5.2f}%".format(100 * accuracy))

   
if __name__ == '__main__':
    main()