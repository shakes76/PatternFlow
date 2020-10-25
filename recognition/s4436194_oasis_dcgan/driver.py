from recognition.s4436194_oasis_dcgan.oasis_dcgan import (
    DCGANModelFramework,
)
from recognition.s4436194_oasis_dcgan.models_helper import *


def run_dcgan_training():
    """

    Returns:

    """
    batch_size = 16
    epochs = 10

    discriminator, generator, size = make_models_256()

    framework = DCGANModelFramework(discriminator, generator, size)
    framework.train_dcgan(batch_size=batch_size, epochs=epochs)


def run_dcgan_tests():
    """

    Returns:

    """
    discriminator, generator, size = make_models_256()

    framework = DCGANModelFramework(discriminator, generator, size)
    framework.test_dcgan()


if __name__ == '__main__':
    # run_dcgan_training()
    run_dcgan_tests()
