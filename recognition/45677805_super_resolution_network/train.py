from dataset import *
from modules import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

"""loss and accuracy plot"""
def make_plot(history, epoch):

    acc_list = history.history['mean_squared_error']
    loss_list = history.history['loss']
    x_epoch = [i for i in range(epoch)]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    fig.tight_layout(h_pad=5)
    ax1.plot(x_epoch, acc_list, label='mean_squared_error')
    ax1.set_xlabel("number of epoch")
    ax1.set_ylabel("maginitude value")
    ax1.set_title("metrics(mean_squared_error)")
    ax2 = fig.add_subplot(212)
    ax2.plot(x_epoch, loss_list, label='loss')
    ax2.set_xlabel("number of epoch")
    ax2.set_ylabel("maginitude value")
    ax2.set_title("loss")
    plt.show()

"""train the model"""
def train():
    #load the train validation and test dataset
    train_data = get_train()
    val_data = get_validation()

    #resize the train , validation and test data set and prefetc the data into CPU
    resized_train_data = resize_img_flow(train_data, 32)
    resized_train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    resized_val = resize_img_flow(val_data, 32)
    resized_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # get the srcnn model
    model = get_model()

    # loss function
    loss_fun = MeanSquaredError()
    optimizer_Adam = Adam(0.001)

    # loss function
    loss_fun = MeanSquaredError()
    optimizer_Adam = Adam(0.001)

    model.compile(loss=loss_fun, optimizer=optimizer_Adam, metrics=['MeanSquaredError'])

    epoch = 1
    history = model.fit(resized_train_data, epochs=epoch, validation_data=resized_val)

    # model.save_weights("G:/Australia/academic/UQ/2022 s2/comp 3710/A3/s4567780 Problem 5  super-resolution network/PatternFlow/recognition/45677805_super_resolution_network/model//modeH5_{}.h5".format(epoch+148))
    make_plot(history, epoch)
train()