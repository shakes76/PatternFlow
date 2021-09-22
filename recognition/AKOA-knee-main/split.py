from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class PathTF:
    def __init__(self, path_list):
        self.path_ds = tf.data.Dataset.from_tensor_slices(path_list)

    def __len__(self):
        return len(self.path_ds)

    def split_iter(self):
        n = len(self)
        batch_size = round(n/3)
        # Shuffle
        path_ds = self.path_ds.shuffle(n)
        return path_ds.batch(batch_size)


def preprocess_image(image, width=224, height=224):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [width, height])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


class Dataset:
    def __init__(self, left_paths, right_paths):
        left = PathTF(left_paths)
        right = PathTF(right_paths)
        self.left_path_ds = [ds for ds in left.split_iter()]
        self.right_path_ds = [ds for ds in right.split_iter()]

    def __getitem__(self, index):
        left_xs = self.left_path_ds[index]
        left_ys = tf.zeros(len(left_xs), dtype=tf.int64)
        right_xs = self.right_path_ds[index]
        right_ys = tf.ones(len(right_xs), dtype=tf.int64)
        xs = tf.concat((left_xs, right_xs), axis=0)
        ys = tf.concat((left_ys, right_ys), axis=0)
        ys = tf.one_hot(ys, depth=2)
        paths = tf.data.Dataset.from_tensor_slices(xs)
        image_ds = paths.map(load_and_preprocess_image)
        label_ds = tf.data.Dataset.from_tensor_slices(ys)
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        return image_label_ds

    def finish(self, image_label_ds, batch_size):
        # Set a shuffle buffer size that is consistent with the size of the dataset to ensure the data
        # It's completely shuffled.
        n = len(image_label_ds)
        ds = image_label_ds.shuffle(buffer_size=n)
        #ds = ds.repeat()
        ds = ds.batch(batch_size)
        # When the model is in training, ` prefetch 'makes the dataset get batch in the background.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds


def get_dataset(data_root, batch_size):
    data_root = Path(data_root)
    left_paths = [path.as_posix()
                  for path in data_root.rglob('**/left/**/*png')]
    right_paths = [path.as_posix()
                   for path in data_root.rglob('**/right/**/*png')]
    loader = Dataset(left_paths, right_paths)
    train, val, test = loader[0], loader[1], loader[2]
    trainset = loader.finish(train, batch_size)
    valset = loader.finish(val, batch_size)
    testset = loader.finish(test, batch_size)
    return trainset, valset, testset
