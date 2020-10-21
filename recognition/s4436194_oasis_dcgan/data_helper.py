import numpy as np
from PIL import Image
from typing import List


class Dataset:
    """
    Class used to read raw data and return in a batch format.
    """

    def __init__(self, data_files: List[str], image_width: int, image_height: int):

        self.data_files = data_files
        self.n_files = len(data_files)

        self.image_width = image_width
        self.image_height = image_height

    def get_batches(self, batch_size: int) -> np.ndarray:
        """
        Serve data batches, yield so this can be called as a loop

        Args:
            batch_size: The first dimension of the output array representing the batch size

        Yields:
            (batch_size, self.image_width, self.image_height, 1) numpy array representing a batch
        """
        idx = 0
        while idx + batch_size <= self.n_files:
            batch = self._get_batch(self.data_files[idx: idx + batch_size])
            idx += batch_size
            yield -1 + (batch - batch.min()) / (batch.max() - batch.min()) * 2

    def _get_batch(self, image_files: List[str]) -> np.ndarray:
        """
        Return an array of images in the form of a batch

        Args:
            image_files: List of file paths of images to read

        Returns:
            (batch_size, self.image_width, self.image_height, 1) numpy array representing batch
        """

        images = [self._get_image(sample_file) for sample_file in image_files]
        batch = np.array(images).astype(np.float32)

        # Add channel dimension
        batch = batch.reshape(batch.shape + (1,))

        return batch

    def _get_image(self, image_path: str) -> np.ndarray:
        """
        Returns the numpy array of a single image. Also responsible for resizing the image appropriately
        using PIL image resize

        Args:
            image_path: Single path to image

        Returns:
            (self.image_width, self.image_height) numpy array representing an image

        """
        image = Image.open(image_path).resize((self.image_width, self.image_height))
        assert image.size == (self.image_width, self.image_height), f"Inconsistent image size: {image.size}"

        return np.array(image.convert(mode="L"))
