import tensorflow as tf
import numpy as np
import PIL
import pathlib


class ImgLoader:

    def __init__(self, data_dir):
        """
        Initialization of ImgLoader object
        :param data_dir: The directory of the dataset
        """
        data_root = pathlib.Path(data_dir)
        self.all_image_paths = list(data_root.glob('*.png'))
        self.all_image_paths = [str(path) for path in self.all_image_paths]

    def load_to_tensor(self, buffer_size, batch_size=256, img_size=64):
        """
        Transfer the dataset to tensor.
        :param buffer_size: The buffer size for the tensor. When 0 is the length of whole dataset
        :param batch_size: The batch size for the tensor. Default value is 256
        :param img_size: Compress images to img_size x img_size. Default value = 64
        :return: A Tensor of the dataset.
        """
        data_tensor = 0
        if buffer_size == 0:
            buffer_size = len(self.all_image_paths)
        train_images = np.ones((buffer_size, img_size, img_size), dtype='float16')
        for i in range(buffer_size):
            img = PIL.Image.open(self.all_image_paths[i]).resize((img_size, img_size))
            img_np = (np.array(img).astype('float16') - 127.5) / 127.5
            train_images[i] = img_np
        train_images = train_images.reshape([buffer_size, img_size, img_size, 1])
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
        return train_dataset
