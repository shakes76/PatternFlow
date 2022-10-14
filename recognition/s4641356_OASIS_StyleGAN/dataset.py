import os
from PIL import Image
import tensorflow as tf
import numpy as np
import GANutils

class OASIS_loader():
    """
    Data loader and preprocessor for local OASIS dataset
    """

    CACHE = "cache/"

    IMAGE_RES = 256
    CROP_PIXELS = 35 #number of blackspace pixels to crop off per side

    def __init__(self, filepath: str = "images/" , resolution = 256) -> None:
        """
        Instantiates a new OASIS data loader and preprocessor for images stored
        at the provided directory

        Args:
            filepath (str, optional): Directory of folder containing OASIS images. Defaults to "images/".
            resolution (int, optional): side length of images to be returned (images will be resized before normalisation and caching). 
                                        WARNING, this will be ignored if a cache is found, and instead the resolution of the images inside the found cache will be used.
                                        Defaults to 256.
        """
        
        self._filepath = filepath
        self._image_list = os.listdir(self._filepath)
        self._num_images_availible = len(self._image_list)
        
        print("Found {} Images".format(self._num_images_availible))

        #check for prexisting processed data
        if os.path.exists(OASIS_loader.CACHE) and (len(os.listdir(OASIS_loader.CACHE)) == self._num_images_availible):
            print("Found existing cache, shall use this")

        else:
            print("Normalising...".format(self._num_images_availible))

            #Temporarily load all images to calculate shared mean and normalise.
            image_vector = np.stack([self._load_image(self._filepath + image_name, resolution) for image_name in self._image_list], axis = 0)
            image_vector = self._normalise(image_vector)
            image_vector = np.split(image_vector, self._num_images_availible, axis = 0)
            print("Normalisation Complete, Caching:")

            #Save data to disk for future load to prevent clogging program memory
            GANutils.make_fresh_folder(OASIS_loader.CACHE)
            for i in range(self._num_images_availible):
                np.save(OASIS_loader.CACHE + str(i) + ".npy", image_vector[i])
                print("{}/{}".format(i,self._num_images_availible))
            
            # #cache mean to prevent needing to recalculate in future
            # np.save(OASIS_loader.MEAN_CACHE,np.array([self._normalisation_mean]))

            print("Data is prepped and ready to go!")

        self._pointer = 0 #Keep a track of how many images we have already used to allow batching

        # self._normalisation_mean = 0 #TODO disable centering

    def get_data(self, num_images: int) -> tf.Tensor:
        """
        Returns a normalized Tensor containing greyscale image pixel intensities of the next availible images.
        Images are sampled in the order they appear in the folder, cyclicly.

        Args:
            num_images (int): size of data vector to return

        Raises:
            ValueError: If the requested number of images is greater than the amount of images availible

        Returns:
            tf.Tensor: Normalised data vector.
        """
        if num_images > self._num_images_availible:
            raise ValueError("Requested more images than availible: \n ({} > {})".format(num_images,self.num_images_availible))

        data = []
        for i in range(num_images):
            data.append(np.load(OASIS_loader.CACHE + str(self._pointer) + ".npy"))
            self._pointer += 1

            #Cycle back to first image to allow multiple epochs
            if self._pointer >= self._num_images_availible:
                self._pointer = 0
            
        return tf.convert_to_tensor(np.concatenate(data, axis = 0))


    # def get_mean(self) -> float:
    #     """
    #     Returns the mean used to normalise data, which can be used to denormalise generated samples

    #     Returns:
    #         float: Group mean used to center normalised data
    #     """
    #     return self._normalisation_mean

    def get_num_images_availible(self) -> int:
        """
        Returns number of availible images

        Returns:
            int: number of images availible to load
        """
        return self._num_images_availible

    def _load_image(self, filepath: str, resolution: int = 256) -> np.array:
        """
        Load a specified image and convert it into a numpy array.
        As the OASIS images are colourless, we convert them to greyscale
        and represent them as a one channel matrix of pixel intensities.
        This significantly improves model training rate for minimal information loss.

        Args:
            filepath (str): Filepath of image to load
            resolution (int, Optional): 

        Returns:
            np.array: Matrix of greyscale image pixel intensities (0,255)
        """
        im = Image.open(filepath).convert('L').resize((resolution,resolution), box = (OASIS_loader.CROP_PIXELS,OASIS_loader.CROP_PIXELS,OASIS_loader.IMAGE_RES-OASIS_loader.CROP_PIXELS,OASIS_loader.IMAGE_RES-OASIS_loader.CROP_PIXELS))# we heavily compress the image to ruin the NN on a local machine. This is feasible by the simplicity of image.
        return np.asarray(im,dtype = np.uint8)[:,:,np.newaxis] #convolutional layers require a channel dimension

            
    def _normalise(self, image_vector: np.array) -> np.array:
        """
        Normalises and centers vector of greyscale image intensity matrices

        Args:
            image_vector (np.array): tensor of image data to normalize
        Returns:
           np.array: Normalised data vector
        """
        # normed_vector = image_vector/255
        # mean = np.mean(normed_vector)
        # return (normed_vector - mean), mean
        return image_vector/255

