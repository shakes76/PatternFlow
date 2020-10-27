from recognition.s4436194_oasis_dcgan.oasis_dcgan import (
    DCGANModelFramework,
    Dataset,
)
from recognition.s4436194_oasis_dcgan.models_helper import *

MODELS_MAP = {
    28: make_models_28(),
    64: make_models_64(),
    128: make_models_128(),
    256: make_models_256(),
}


def run_dcgan_training(res):
    """

    Returns:

    """

    assert res in MODELS_MAP, f"Resolution must be either 28, 64, 128, or 256: {res}"

    batch_size = 16
    epochs = 10

    discriminator, generator, size = MODELS_MAP[res]

    framework = DCGANModelFramework(discriminator, generator, size)
    framework.train_dcgan(batch_size=batch_size, epochs=epochs)


def run_dcgan_tests(res):
    """

    Returns:

    """
    assert res in MODELS_MAP, f"Resolution must be either 28, 64, 128, or 256: {res}"
    discriminator, generator, size = MODELS_MAP[res]

    framework = DCGANModelFramework(discriminator, generator, size)
    framework.test_dcgan()


if __name__ == '__main__':

    res = 28

    run_dcgan_training(res)
    # run_dcgan_tests(res)
