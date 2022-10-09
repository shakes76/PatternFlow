from json import load
from modules import siamese
from dataset import load_train_data


def train():
    model = siamese(128, 128)

    train, val = load_train_data()

    train = train.batch(32)
    val = val.batch(32)

    model.fit(train, epochs=20, validation_data=val)

    predict(val, model)


def predict(ds, model):
    for pair, label in ds:
        pred = model.predict(pair)
        for i in range(len(pred)):
            print(pred[i], label[i])
        break 

train()