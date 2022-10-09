import os
from PIL import Image
from tensorflow import Tensor, convert_to_tensor, reduce_mean
from numpy import asarray, ndarray, uint8

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


    def get_data(self, num_images: int) -> tuple[Tensor, float]:
        """
        Returns a normalized Tensor containing greyscale image pixel intensities.
        Images appear in order they appear in folder, with images being loaded into
        memory for future repeat usage.

        Args:
            num_images (int): size of data vector to return

        Raises:
            ValueError: If the requested number of images is greater than the amount of images in the specified folder

        Returns:
            tuple[Tensor, float]: Normalised data vector, Group mean used to center normalised data
        """
        if num_images > self._num_images_availible:
            raise ValueError("Requested more images than availible: \n ({} > {})".format(num_images,self.num_images_availible))

        #Load in any additional images required to meet desired amount
        while len(self._cache) < num_images:
            self._cache.append(self._load_image(self._filepath + self._image_list[len(self._cache)]))

        return self._normalise(convert_to_tensor(self._cache[:num_images]))

    def _load_image(self, filepath: str) -> ndarray:
        """
        Load a specified image and convert it into a numpy array.
        As the OASIS images are colourless, we convert them to greyscale
        and represent them as a one channel matrix of pixel intensities.
        This significantly improves model training rate for minimal information loss.

        Args:
            filepath (str): Filepath of image to load

        Returns:
            ndarray: Matrix of greyscale image pixel intensities (0,255)
        """
        im = Image.open(filepath).convert('L')
        return asarray(im,dtype=uint8)

            
    def _normalise(self, image_vector: Tensor) -> tuple[Tensor, float]:
        """
        Normalises and centers vector of greyscale image intensity matrices

        Returns:
           tuple[Tensor, float]: Normalised data vector, Group mean used to center normalised data
        """
        normed_vector = image_vector/255
        mean = reduce_mean(normed_vector)
        return (normed_vector - mean), mean

        

test = OASIS_loader()
print(test.get_data(10))
