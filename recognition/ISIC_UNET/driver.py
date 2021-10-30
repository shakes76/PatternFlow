import tensorflow as tf
import os
from recognition.ISIC_UNET.ImprovedUnet import *
from recognition.ISIC_UNET.UnetData import *
import matplotlib.pyplot


# https://www.fatalerrors.org/a/dice-loss-in-medical-image-segmentation.html
def dsc_co(ytrue, ypred):
    # calculate the dice co-efficient
    smooth = 1
    true_flat = tf.keras.backend.flatten(ytrue)
    pred_flat = tf.keras.backend.flatten(ypred)
    inter = tf.keras.backend.sum(true_flat * pred_flat)
    return 2 * (inter + smooth) / (tf.keras.backend.sum(true_flat) + tf.keras.backend.sum(pred_flat) + smooth)


def dsc_loss(ytrue, ypred):
    # calculate the dice co-efficient loss
    return 1 - dsc_co(ytrue, ypred)


def show_pred(model, final_test, num):
    # shows predictions from the final model
    print("Showing predictions")
    matplotlib.pyplot.figure(figsize=(4 * 4, num * 4))
    i = 0
    for image, mask in final_test.take(num): # plots the images and the masked images next to the predicted images
        predictions = model.predict(image[tf.newaxis, ...])[0]
        matplotlib.pyplot.subplot(num, 4, 4 * i + 1)
        matplotlib.pyplot.imshow(image)
        matplotlib.pyplot.subplot(num, 4, 4 * i + 2)
        matplotlib.pyplot.imshow(mask[:, :, 0], cmap='gray')
        matplotlib.pyplot.subplot(num, 4, 4 * i + 3)
        matplotlib.pyplot.imshow(predictions[:, :, 0], cmap='gray')
        i = i + 1
    matplotlib.pyplot.savefig('predictions.png')
    matplotlib.pyplot.show()


def plot_graphs(history, evaluation):
    # saves and plots the graphs for the accuracy, loss and dice-coefficient. Also saves the evaluation text and history
    # text for easier viewing when debugging

    # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
    fig, axs = matplotlib.pyplot.subplots(3)
    # plot the accuracy
    axs[0].plot(history.history['accuracy'], label='training accuracy', color="b")
    axs[0].plot(history.history['val_accuracy'], label='validation accuracy', color="g")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")

    # plot loss
    axs[1].plot(history.history['loss'], label='training loss', color='r')
    axs[1].plot(history.history['val_loss'], label='validation loss', color='c')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")

    # plot dice co-efficient
    axs[2].plot(history.history['dsc_co'], label='training dsc', color='m')
    axs[2].plot(history.history['val_dsc_co'], label='validation dsc', color='y')
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Dice Co-efficient")
    fig.tight_layout()
    fig.legend(loc='right')

    fig.set_size_inches(15, 15)
    matplotlib.pyplot.savefig('History.png')
    matplotlib.pyplot.show()

    # writes to files
    with open('history.txt', 'w') as f:
        f.write(str(history.history))

    with open('evaluation.txt', 'w') as f:
        f.write(str(evaluation))


if __name__ == '__main__':
    # see if there are any gpus that can be used
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    # sets the max amount of images
    max_images = 1800

    # https://stackoverflow.com/questions/52123670/neural-network-converges-too-fast-and-predicts-blank-results
    # sets the learning rate
    learning_rate = 3*10**(-5)

    # path to the images
    imagePath = "C:\\Users\\Mark\\Desktop\\comp3710 report\\PatternFlow\\recognition\\ISIC_UNET\\ISIC2018_Task1-2_Training_Data"

    # prepare a given number of images
    final_train, final_test, final_val = prepare_images(max_images, imagePath)

    # define the model and return it
    model = IUNET_model()

    # compile the model with a specified learning rate, loss and given metrics
    model.compile(tf.keras.optimizers.Adam(lr=learning_rate), loss=dsc_loss, metrics=['accuracy', dsc_co])
    model.summary()  # gives a summary of the model to help with debugging

    # training the model with a given batch size and number of epochs
    print("Training model")
    history = model.fit(final_train.batch(8), epochs=15, validation_data=final_val.batch(8))

    # evaluates the model with a given batch size
    print("Evaluating model")
    evaluation = model.evaluate(final_test.batch(8))

    # calls the function which plots all the graphs
    plot_graphs(history, evaluation)

    # pass model, the testing data and batch size to see predictions
    show_pred(model, final_test, 3)
