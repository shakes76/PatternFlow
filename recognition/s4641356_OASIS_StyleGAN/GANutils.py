from PIL import Image
import numpy as np
from numpy import ndarray, uint8

class GANUtilities():
    """
    Various helper functions for use with and around GAN and its output.
    Packaged into a class due to some import issues
    """
    def denormalise(data: ndarray, mean: float) -> ndarray:
        """
        Take output of GAN and undo data preprocessing procedure to return
        a matrix of integer greyscale image pixel intensities suitable to
        convert directly into an image.

        Args:
            data (ndarray): normalised data
            mean (float): grop mean used to centralise original normalised data

        Returns:
            ndarray: denormalized data
        """
        data = np.array(data) #cast to ndarray from any array like (Allows Tensor Compatibility)
        decentered = data + mean
        return (decentered * 255).astype(uint8)


    def create_image(data: ndarray, name: str = None, output_folder: str = "output/") -> Image:
        """
        Creates an new PNG image from a generated data matrix.
        Saves image to output folder if a name is specified

        Args:
            data (ndarray): uint8 single channel matrix of greyscale image pixel intensities
            name (str or NoneType, optional): name to save image as in output folder, If None image is not saved. Defaults to None.
            output_folder (str, optional): path of output folder. Defaults to "output/".

        Returns:
            Image: Generated image
        """
        im = Image.fromarray(data).convert("RGBA")
        if not name == None:
            im.save(output_folder+name+".png")
        return im

