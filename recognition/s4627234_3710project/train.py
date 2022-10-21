import dataset
import modules
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


def training(data_reshape = False):
    """ training data with model in modules.py with data from dataset.py

    Args:
        data_reshape (bool, optional): true if wants to reshape data. Defaults to False.

    Returns:
        model, history: return the model and trained history of data
    """
    train_x, train_y, test_x, test_y = dataset.load_dataset(data_reshape)
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    model = modules.UNet_imp()

    # fit the model with normal learning rate
    model.compile(optimizer = Adam(0.0005), loss = modules.DSC_loss, metrics=['accuracy', modules.DSC])

    history = model.fit(train_x, train_y,  validation_data= (test_x, test_y),
                            batch_size=8,shuffle='True',epochs=20)

    return model, history


def plot_data(history, type):
    """ plot DSC or DSC loss data and save into images

    Args:
        history (keras.callbacks.History): history of training model
        type (string): decide acc or loss
    """
    plt.figure(figsiz = (10, 5))

    if type == 'acc': add = 'DSC'
    else: add = 'loss'

    plt.plot(history.history['' + add], label='Training ' + add)
    plt.plot(history.history['val_' + add], label='Validation ' + add)
    plt.title('Test vs Validation ' + add)
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('' + add)
    plt.savefig('./img/'+ add +'.png')
    plt.show()


def main():
    """
        Training and save the model
    """

    model, history = training(data_reshape = False)
    model.save("imp_unet_model.h5")

    plot_data(history, 'acc')
    plot_data(history, 'loss')

if __name__ == "__main__":
    main()