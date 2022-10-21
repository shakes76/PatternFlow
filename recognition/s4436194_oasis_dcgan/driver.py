"""
OASIS DCGAN Driver script

Runs training and testing for the main DCGAN implementation in this project. In the main run statement, ensure
that the model size is correctly selected. The model size can only be one of 28, 64, 128, or 256.

NOTE: Running this relies on the data files being in the same folder. If data is not present, the code will fail

@author nthompson97
"""

from recognition.s4436194_oasis_dcgan.models_helper import (
    make_models_28,
    make_models_64,
    make_models_128,
    make_models_256
)
from recognition.s4436194_oasis_dcgan.oasis_dcgan import DCGANModelFramework

MODELS_MAP = {
    28: make_models_28(),
    64: make_models_64(),
    128: make_models_128(),
    256: make_models_256(),
}


def run_dcgan_training(resolution):
    """
    Performs training for the DCGAN models. The number of epochs and batch size can be controlled from this
    top level.

    Args:
        resolution: The size of the output image
    """

    assert resolution in MODELS_MAP, f"Resolution must be either 28, 64, 128, or 256: {resolution}"
    discriminator, generator, resolution = MODELS_MAP[resolution]

    batch_size = 16
    epochs = 10

    framework = DCGANModelFramework(discriminator, generator, resolution)
    framework.train_dcgan(batch_size=batch_size, epochs=epochs)


def run_dcgan_tests(resolution):
    """
    Builds a model and displays an image displayed from that model's generator. This must be a preexisting model.

    Args:
        resolution: The size of the output image
    """
    assert resolution in MODELS_MAP, f"Resolution must be either 28, 64, 128, or 256: {resolution}"
    discriminator, generator, resolution = MODELS_MAP[resolution]

    framework = DCGANModelFramework(discriminator, generator, resolution)
    framework.test_dcgan(save_dir="2020-10-27-64x64")


if __name__ == '__main__':
    size = 28

    run_dcgan_training(size)
    run_dcgan_tests(size)
