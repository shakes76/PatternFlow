from recognition.s4436194_oasis_dcgan.oasis_dcgan import (
    DCGANModelFramework,
)


def run_dcgan_training():
    batch_size = 16
    epochs = 10

    framework = DCGANModelFramework()
    framework.train_dcgan(batch_size=batch_size, epochs=epochs)


def run_dcgan_tests():
    framework = DCGANModelFramework()
    framework.test_dcgan()


if __name__ == '__main__':
    run_dcgan_training()
    run_dcgan_tests()
