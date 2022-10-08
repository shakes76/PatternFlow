from json import load
from modules import siamese
from dataset import load_train_data


def train():
    model = siamese(128, 128)

    train, val = load_train_data()

    train = train.batch(16)
    val = val.batch(16)

    model.fit(train, epochs=20, validation_data=val)
