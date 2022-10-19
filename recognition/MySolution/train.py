from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import keras.losses


def plot_loss_epoch(history):
    """
    The plot_loss_epoch() function takes the history of the trained model
    and create a plot for loss vs epoch. The plot is saved as a png file

    Args:
        history: The history to use and plot

    """
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig("pretraining_plot.png")
    plt.show()


def plot_all(history):
    """
    The plot_all() function takes the history of the trained model
    and create two plots for accuracy and loss of both training and validation.
    The plots are saved as png files

    Args:
        history: The history to use and plot

    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("pretrained_plot_accuracy.png")
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("pretrained_plot_loss.png")
    plt.show()


def handle_training(module):
    """
    The handle_training() function takes the module and run the models
    to train against the training datasets. The model will first be trained
    to handle feature recognitions and loss. Then the model will be trained
    against the dataset for prediction. Both models will be saved and the
    accuracy and loss graphs will be created by calling the plotting functions.
    The first model training will use 100 epoch while the second will use 200
    with patience of 20 epochs

    Args:
        module: The module class to access model values

    """
    train_gen = module.get_train_gen()
    model = module.get_model()
    history_log = keras.callbacks.CSVLogger(
        "history_log.csv",
        separator=",",
        append=True
    )
    history = model.fit(
        train_gen,
        epochs=100,
        verbose=2,
        callbacks=[history_log]
    )
    model.save("pre-trained_model")
    plot_loss_epoch(history.history)
    new_model = module.model_retrain(module.get_data_group())
    prediction = EarlyStopping(
        monitor="val_acc", patience=20, restore_best_weights=True
    )
    train_gen, test_gen, val_gen = module.get_gen()
    pretrained_history = new_model.fit(
        train_gen,
        epochs=200,
        verbose=2,
        validation_data=module.val_gen,
        callbacks=[prediction],
    )
    plot_all(pretrained_history)
    new_model.save("finalised_model")








