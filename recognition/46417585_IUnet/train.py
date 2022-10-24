import matplotlib.pyplot as plt
from keras.optimizers import Adam

from constants import EPOCHS
from dataset import train_data, validation_data
from modules import UNet
from utils import DSC, DSC_loss


def train():
    model = UNet()

    model.compile(optimizer=Adam(learning_rate=0.0003), loss=DSC_loss, metrics=[DSC])

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
    )

    model.save("UNet.h5")

    return model


if __name__ == "__main__":
    train()
