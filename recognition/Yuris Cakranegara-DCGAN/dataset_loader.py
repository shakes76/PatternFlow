import os.path
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageDatasetLoader:
    def __init__(self, dir_data, img_shape, color_mode="grayscale"):
        """Initialize an image dataset loader.

        Args:
            dir_data (str): The directory of the dataset. Has to end with a forward slash. Example: "home/"
            img_shape (tuple): The shape of the image. Has to be a 3 tuple.
            color_mode (str, optional): The color scheme of images loaded. Defaults to "grayscale".
        """
        assert dir_data[-1] == "/", \
            "Data directory has to end with a forward slash (/)."
        assert type(img_shape) == tuple, \
            "Image shape has to be a 3 tuple."
        assert len(img_shape) == 3, \
            "Image shape has to be a 3 tuple."
        assert color_mode in ["grayscale", "rgb", "rgba"], \
            "Color mode has to be either grayscale, rgb, or rgba."
        
        self.dir_data = dir_data
        self.img_shape = img_shape
        self.color_mode = color_mode

    def load_data(self, nm_images=None):
        """Loads the image dataset from the specified directory. The number of images loaded 
        will be specified by the argument nm_images.

        Args:
            nm_images (int, optional): The number of images to be loaded. Defaults to load 
            all images in the directory.

        Returns:
            np.ndarray: The images list in the form of a numpy array.
        """

        imgs = np.sort(os.listdir(self.dir_data))
        if (nm_images != None):
            assert type(nm_images) == int, \
                "nm_images has to be an integer."
            imgs = imgs[:nm_images]

        X = []
        for _i, myid in enumerate(imgs):
            image = load_img(self.dir_data + myid, target_size=self.img_shape, color_mode=self.color_mode)
            image = img_to_array(image, dtype="uint8")
            X.append(image)
        X = np.array(X)
        return X
