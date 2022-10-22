from dataset import *
from modules import *

import tensorflow_addons as tfa
import matplotlib.pyplot as plt

num_classes = 2
#input_shape = (128, 128, 3)
batch_size = 16
epochs = 60
learning_rate = 0.001
weight_decay = 0.0001

patch_size = 2
patch_count = (128 // patch_size) ** 2
proj_dims = 256
latent_dims = 256
forward_units = [
    proj_dims,
    proj_dims
]
head_count = 8
num_t_blocks = 4
dropout_rate = 0.2
num_iters = 2
classifier_units = [
    proj_dims,
    num_classes,
]

REQ_DIR = "../content/drive/MyDrive/Colab Notebooks/Report/AD_NC"

def run_model(model):
    
    x_train, y_train, x_test, y_test = process_image(REQ_DIR)

    optimizer = tfa.optimizers.LAMB(
        learning_rate=learning_rate, weight_decay_rate=weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )

    # reduce learning rates as model progresses
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # fit the model
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[reduce_lr]
    )

    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    model.save('complete_model')

    return history




def main():

    perceiver_classifier = Perceiver(
        patch_size,
        patch_count,
        latent_dims,
        proj_dims,
        head_count,
        num_t_blocks,
        forward_units,
        dropout_rate,
        num_iters,
        classifier_units
    )

    history = run_model(perceiver_classifier)


    # plot results
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('my_plot.png')

if __name__ == "__main__":
    main()