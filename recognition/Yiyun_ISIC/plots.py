"""Generate plots for the model
   Author: Yiyun Zhang (s4513350)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_predictions(model, test_dataset, plot_batch=4, threshold=0.5):
    """Plot predictions of a model on a test dataset.

    Args:
        model (Model): Model to be evaluated.
        test_dataset (Dataset): Test dataset.
        plot_batch (int, optional): Number of examples to be plotted. Defaults to 4.
        threshold (float, optional): Threshold for binarisation. Defaults to 0.5.
    """
    # generate predictions
    test_images, test_labels = next(iter(test_dataset.batch(plot_batch)))
    predictions = model.model.predict(test_images)

    # initialise figure
    figure = plt.figure(constrained_layout=True, figsize=(14, 12))
    # add padding to the rows
    figure.set_facecolor('white')
    spec = gridspec.GridSpec(ncols=4, nrows=plot_batch,
                             figure=figure, hspace=0.1, wspace=0.1)

    for i in range(plot_batch):
        # plot original image
        ax1 = figure.add_subplot(spec[i, 0])
        ax1.imshow(test_images[i])
        ax1.set_title("Actual Image")
        ax1.axis('off')

        # plot ground truth
        ax2 = figure.add_subplot(spec[i, 1])
        ax2.imshow(test_labels[i], cmap='binary')
        ax2.set_title("Ground Truth")
        ax2.axis('off')

        # plot prediction with threshold
        ax3 = figure.add_subplot(spec[i, 2])
        ax3.imshow(predictions[i] > threshold, cmap='binary')
        ax3.set_title("Prediction (threshold = {})".format(threshold))
        ax3.axis('off')

        # plot prediction
        ax4 = figure.add_subplot(spec[i, 3])
        ax4.imshow(predictions[i], cmap='Blues')
        ax4.set_title("Prediction (continous)")
        ax4.axis('off')

    figure.savefig("./images/plot.png")
    figure.show()


def plot_metrics(history):
    """Plot loss, accuracy and dice coefficient of a model.

    Args:
        history (History): Model training history.
    """
    figure = plt.figure(constrained_layout=True, figsize=(14, 3))
    figure.set_facecolor('white')
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=figure, wspace=0.1)

    # summarise history for accuracy
    ax1 = figure.add_subplot(spec[0, 0])
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['training set', 'validation set'], loc='lower right')

    # summarise history for dice coefficient
    ax2 = figure.add_subplot(spec[0, 1])
    ax2.plot(history.history['dice_coef'])
    ax2.plot(history.history['val_dice_coef'])
    ax2.set_title('Model Dice Coefficient')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_xlabel('Epoch')
    ax2.legend(['training set', 'validation set'], loc='lower right')

    # summarise history for loss
    ax3 = figure.add_subplot(spec[0, 2])
    ax3.plot(history.history['loss'])
    ax3.plot(history.history['val_loss'])
    ax3.set_title('Model Loss')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Epoch')
    ax3.legend(['training set', 'validation set'], loc='upper right')

    figure.savefig("./images/plot_metrics.png")
    figure.show()
