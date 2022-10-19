import tensorflow as tf


HEIGHT = 256
WIDTH = 256

training_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Part1_GroundTruth"
test_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Test_v2_Part1_GroundTruth"
validation_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Validation_Part1_GroundTruth"
training_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Data"
validation_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Validation_Data"
test_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Test_v2_Data"

seed = 150
rng = tf.random.Generator.from_seed(seed, alg='philox')


def augment_train_data(data_filenames, mask_filenames):
    image = preprocess_data(data_filenames)
    mask = preprocess_labels(mask_filenames)
    new_seed = rng.make_seeds(2)[0]
    im_seed = tf.random.experimental.stateless_split(new_seed, num=1)[0, :]
    image = tf.image.stateless_random_flip_left_right(image, seed=im_seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=im_seed)
    image = tf.image.stateless_random_flip_up_down(image, seed=im_seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed=im_seed)
    image = tf.image.stateless_random_saturation(image, 1, 3, seed=im_seed)
    return image, mask


def preprocess_data(filenames):
    data = tf.io.read_file(filenames)
    image = tf.io.decode_jpeg(data, channels=3)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess_labels(filenames):
    data = tf.io.read_file(filenames)
    mask = tf.io.decode_png(data, channels=1)
    mask = tf.image.resize(mask, [HEIGHT, WIDTH])
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return mask


def data_loader():
    # input data folders

    training_data = tf.data.Dataset.list_files(training_input_dir, shuffle=False)
    test_data = tf.data.Dataset.list_files(test_input_dir, shuffle=False)
    test_data = test_data.map(preprocess_data)
    validation_data = tf.data.Dataset.list_files(validation_input_dir, shuffle=False)
    validation_data = validation_data.map(preprocess_data)

    training_labels = tf.data.Dataset.list_files(training_labels_dir, shuffle=False)
    test_labels = tf.data.Dataset.list_files(test_labels_dir, shuffle=False)
    test_labels = test_labels.map(preprocess_labels)
    validation_labels = tf.data.Dataset.list_files(validation_labels_dir, shuffle=False)
    validation_labels = validation_labels.map(preprocess_labels)

    train_ds = tf.data.Dataset.zip((training_data, training_labels))
    test_ds = tf.data.Dataset.zip((test_data, test_labels))
    validation_ds = tf.data.Dataset.zip((validation_data, validation_labels))
    train_ds = train_ds.map(augment_train_data)

    return train_ds, test_ds, validation_ds
