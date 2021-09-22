import argparse
import tensorflow as tf

from split import get_dataset
from model_create import create_model, plot

parser = argparse.ArgumentParser()
parser.add_argument("data_root")
#parser.add_argument("batch_size")
args = parser.parse_args()


if __name__ == "__main__":
    data_root = args.data_root
    batch_size = 32 #args.batch_size
    trainset, valset, testset = get_dataset(data_root, batch_size)
    model = create_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        trainset,
        epochs=10,
        validation_data=valset,
        validation_steps=10
    )
    loss, accuracy = model.evaluate(testset)
    print(f'Test accuracy : {accuracy}, loss {loss}', )
    plot(history)
