import tensorflow as tf
from train import train_model

if __name__ == "__main__":
    model, dataset = train_model()

    pred = model.predict(dataset.testing)

