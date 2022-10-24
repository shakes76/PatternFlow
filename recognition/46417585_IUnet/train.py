from constants import EPOCHS
from dataset import train_data, validation_data
from modules import UNet
from utils import DSC, DSC_loss


def train():
    model = UNet()

    model.compile(optimizer="adam", loss=DSC_loss, metrics=[DSC])

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
    )

    return model, history
