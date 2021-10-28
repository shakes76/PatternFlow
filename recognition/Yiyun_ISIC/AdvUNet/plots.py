import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_predictions(model, test_dataset, plot_batch=4, threshold=0.5):
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
