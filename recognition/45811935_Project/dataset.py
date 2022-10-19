"""
    Loads and preprocesses all data.

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_preprocess_image_data(path, img_dim, batch_size, validation_split=None, subset=None,
                               seed=None, shift=0):
    """
        Load and preprocess image data (in our case, the ADNI dataset).

        Args:
            path: absolute path to image data (unzipped).
            img_dim: Size of images (assume square)
            batch_size: Size of each batch
            validation_split: proportion of data to keep for validation set (None by default -
                              i.e. no train/validation split)
            subset: Whether to load train or validation set if using train/validation split (None by
                    default - i.e. no train/validation split)
            seed: Random seed to ensure no data leakage due to train/validation split (None by
                  default - i.e. no train/validation split)
            shift: amount to shift each pixel (0 by default)

        Returns:
            A tf.data.Dataset of the image files.
    """
    img_data = image_dataset_from_directory(path,
                                            label_mode=None,
                                            image_size=(img_dim, img_dim),
                                            color_mode="rgb",
                                            batch_size=batch_size,
                                            shuffle=True,
                                            validation_split=validation_split,
                                            subset=subset,
                                            seed=seed)

    return img_data.map(lambda x: (x / 255.0) - shift)
