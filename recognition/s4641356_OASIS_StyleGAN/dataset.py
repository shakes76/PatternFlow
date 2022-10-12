import os
from PIL import Image
import tensorflow as tf
import numpy as np

class OASIS_loader():
    """
    Data loader and preprocessor for local OASIS dataset
    """
    def __init__(self, filepath: str = "images/") -> None:
        """
        Instantiates a new OASIS data loader and preprocessor for images stored
        at the provided directory

        Args:
            filepath (str, optional): Directory of folder containing OASIS images. Defaults to "images/".
        """
        self._filepath = filepath
        self._image_list = os.listdir(self._filepath)
        self._num_images_availible = len(self._image_list)
        self._cache = []
        print("Found {} Images".format(self._num_images_availible))


    def get_data(self, num_images: int) -> tuple[tf.Tensor, float]:
        """
        Returns a normalized Tensor containing greyscale image pixel intensities.
        Images appear in order they appear in folder, with images being loaded into
        memory for future repeat usage.

        Args:
            num_images (int): size of data vector to return

        Raises:
            ValueError: If the requested number of images is greater than the amount of images in the specified folder

        Returns:
            tuple[tf.Tensor, float]: Normalised data vector, Group mean used to center normalised data
        """
        if num_images > self._num_images_availible:
            raise ValueError("Requested more images than availible: \n ({} > {})".format(num_images,self.num_images_availible))

        #Load in any additional images required to meet desired amount
        while len(self._cache) < num_images:
            #wrap returned array in an additioal axis and append to list of lists
            self._cache.append(self._load_image(self._filepath + self._image_list[len(self._cache)]))

        data,mean = self._normalise(np.stack(self._cache[0:num_images], axis = 0))
        return tf.convert_to_tensor(data),mean

    def _load_image(self, filepath: str) -> np.array:
        """
        Load a specified image and convert it into a numpy array.
        As the OASIS images are colourless, we convert them to greyscale
        and represent them as a one channel matrix of pixel intensities.
        This significantly improves model training rate for minimal information loss.

        Args:
            filepath (str): Filepath of image to load

        Returns:
            np.array: Matrix of greyscale image pixel intensities (0,255)
        """
        im = Image.open(filepath).convert('L')
        return np.asarray(im,dtype = np.uint8)[:,:,np.newaxis] #convolutional layers require a channel dimension

            
    def _normalise(self, image_vector: np.array) -> tuple[np.array, float]:
        """
        Normalises and centers vector of greyscale image intensity matrices

        Args:
            image_vector (np.array): tensor of image data to normalize
        Returns:
           tuple[np.array, float]: Normalised data vector, Group mean used to center normalised data
        """
        normed_vector = image_vector/255
        mean = np.mean(normed_vector)
        return (normed_vector - mean), mean

