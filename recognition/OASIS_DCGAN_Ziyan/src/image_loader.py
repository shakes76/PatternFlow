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

    def load_to_tensor(self, target_slice=None, batch_size=256, img_size=64):
        """
        Transfer the dataset to tensor.
        :param target_slice: The target slices for the model training
        :param batch_size: The batch size for the tensor. Default value is 256
        :param img_size: Compress images to img_size x img_size. Default value = 64
        :return: A Tensor of the dataset.
        """
        target_add = self.all_image_paths
        if target_slice:
            target_add = []
            for path_add in self.all_image_paths:
                current_slice = int(path_add.split('_')[-1].split('.')[0])
                if current_slice in target_slice:
                    target_add.append(path_add)
        buffer_size = len(target_add)
        train_images = np.ones((buffer_size, img_size, img_size), dtype='float16')
        for i in range(buffer_size):
            img = PIL.Image.open(target_add[i]).resize((img_size, img_size))
            img_np = (np.array(img).astype('float16') - 127.5) / 127.5
            train_images[i] = img_np
        train_images = train_images.reshape([buffer_size, img_size, img_size, 1])
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
        return train_dataset


if __name__ == '__main__':
    tester = ImgLoader('D:\Datasets\keras_png_slices_data\keras_png_slices_train')
    tester.load_to_tensor()
