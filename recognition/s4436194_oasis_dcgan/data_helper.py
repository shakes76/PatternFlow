import numpy as np
from PIL import Image


class Dataset:
    """
    Class used to read raw data and return in a batch format.
    """

    def __init__(self, data_files, image_width, image_height):
        self.data_files = data_files
        self.n_files = len(data_files)

        self.image_width = image_width
        self.image_height = image_height

    def get_batches(self, batch_size: int) -> np.ndarray:
        idx = 0
        while idx + batch_size <= self.n_files:
            batch = self._get_batch(self.data_files[idx: idx + batch_size])
            idx += batch_size
            yield batch

    def _get_batch(self, image_files: list) -> np.ndarray:
        images = [self._get_image(sample_file) for sample_file in image_files]
        batch = np.array(images).astype(np.float32)

        # Add channel dimension
        batch = batch.reshape(batch.shape + (1,))

        return batch

    def _get_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        assert image.size == (self.image_width, self.image_height), f"Inconsistent image size: {image.size}"

        return np.array(image.convert(mode="L"))
