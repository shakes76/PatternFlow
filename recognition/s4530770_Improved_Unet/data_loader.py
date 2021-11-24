import tensorflow as tf
import pathlib
import os


class DataLoader:
    def __init__(self, path, batch_size=32, validation_size=0.2, test_size=0.1, seed=142):
        """
        This class loads the ISIC data, stores a training, validation and testing dataset.
        Folder must be in format:
            ISIC2018_Task1-2_Training_Data
                |_ ISIC2018_Task1_Training_GroundTruth_x2
                |_ ISIC2018_Task1-2_Training_Input_x2

        :param path: Path to ISIC2018_Task1-2_Training_Data, file separators should be '\\'
        :param batch_size: batch size of dataset.
        :param validation_size: Percentage of dataset to use as validation dataset in decimals, default is 0.2
        :param test_size: Percentage of dataset to use as test dataset in decimals, default is 0.1
        :param seed: Seed for random augmentation on training set. Default is 142.
        """
        self.path = path
        self.batch_size = batch_size
        self.image_shape = [384, 512]
        self.seed = seed

        # Create dataset out of list of input files
        list_ds = tf.data.Dataset.list_files(self.path + "\\ISIC2018_Task1-2_Training_Input_x2\\*.jpg", shuffle=True)

        data_dir = pathlib.Path(self.path)
        image_count = len(list(data_dir.glob('ISIC2018_Task1-2_Training_Input_x2\\*.jpg')))

        # Separate into train, validation and test datasets
        val_size = int(image_count * validation_size)
        test_size = int(image_count * test_size)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)
        test_ds = train_ds.take(test_size)
        train_ds = train_ds.skip(test_size)

        # For stateless seed generation, ensures seed always performs same augmentation
        # on both image and mask.
        rng = tf.random.Generator.from_seed(self.seed, alg='philox')

        @tf.function
        def augment(file_path):
            """
            Maps to training set. Takes image and mask from dataset and applies random
            augmentation on them.

            :param file_path: Input file_path.
            :return: Image and associated mask.
            """
            img, mask = process_path(file_path)
            new_seed = rng.make_seeds(2)[0]
            im_seed = tf.random.experimental.stateless_split(new_seed, num=1)[0, :]
            img = tf.image.stateless_random_flip_left_right(img, seed=im_seed)
            mask = tf.image.stateless_random_flip_left_right(mask, seed=im_seed)
            img = tf.image.stateless_random_flip_up_down(img, seed=im_seed)
            mask = tf.image.stateless_random_flip_up_down(mask, seed=im_seed)
            img = tf.image.stateless_random_saturation(img, 1, 3, seed=im_seed)
            return img, mask

        @tf.function
        def get_mask(file_path):
            """
            Gets associated mask file path from given input file path

            :param file_path: Input image file path.
            :return: Returns mask file path for input image file path.
            """
            parts = tf.strings.split(file_path, os.path.sep)
            file_name = tf.strings.split(parts[-1], '.')[0]
            file_name = file_name + '_segmentation.png'
            return self.path + '\\ISIC2018_Task1_Training_GroundTruth_x2\\' + file_name

        @tf.function
        def process_path(file_path):
            """
            Loads image and mask tensors from file path and preprocesses them.

            :param file_path: Input image file path.
            :return: Image and mask tensor.
            """
            img = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
            mask = tf.io.decode_png(tf.io.read_file(get_mask(file_path)), channels=1)
            img = tf.image.resize(img, self.image_shape)
            mask = tf.image.resize(mask, self.image_shape)
            img = tf.cast(img, tf.float32) / 255.0
            mask = tf.cast(mask, tf.float32) / 255.0

            img = tf.reshape(img, tuple(self.image_shape + [3]))
            mask = tf.reshape(mask, tuple(self.image_shape + [1]))
            return img, mask

        def configure_for_performance(ds):
            """
            Configures given dataset to be shuffled at every iteration and batched into
            given batch size.

            :param ds: tf.data.Dataset to be configured
            :return: Configured dataset.
            """
            ds = ds.shuffle(buffer_size=300, reshuffle_each_iteration=True)
            ds = ds.batch(self.batch_size, num_parallel_calls=self.AUTOTUNE)
            return ds

        self.AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.map(augment, num_parallel_calls=self.AUTOTUNE)
        val_ds = val_ds.map(process_path, num_parallel_calls=self.AUTOTUNE)
        test_ds = test_ds.map(process_path, num_parallel_calls=self.AUTOTUNE)

        self.train_ds = configure_for_performance(train_ds)
        self.val_ds = configure_for_performance(val_ds)
        self.test_ds = configure_for_performance(test_ds)

    def get_training_set(self):
        return self.train_ds

    def get_validation_set(self):
        return self.val_ds

    def get_test_set(self):
        return self.test_ds
