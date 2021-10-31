from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt




def dice_coef(y_true, y_pred, smooth=1):
    """
    Reference: https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


class UnetTrainer:
    def __init__(self, model, data, learning_rate=0.001):
        """
        This class manages the training and prediction of the model. Compiles given
        model with dice coefficient loss and Adam optimiser.
        References:
            [1] https://www.tensorflow.org/tutorials/images/segmentation

        :param model: A tf.keras.Model class
        :param data: A DataLoader class
        :param learning_rate: The learning rate of optimiser. Defaults to 0.001.
        """
        self.model = model
        self.data = data
        self.history = None
        self.learning_rate = learning_rate
        optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimiser,
                           loss=dice_coef_loss,
                           metrics=['accuracy', dice_coef])

    def train(self, epochs):
        """
        Trains model for given epochs, stores history.

        :param epochs: Number of epochs training will run for as int.
        """
        train_ds = self.data.get_training_set()
        val_ds = self.data.get_validation_set()
        self.history = self.model.fit(train_ds,
                                      epochs=epochs,
                                      validation_data=val_ds)

    def show_predictions(self, num=1):
        """
        Predicts mask of images from test set, 'num' times.

        :param num: Number of prediction to make. Defaults to 1.
        """
        dataset = self.data.get_training_set()
        titles = ['Input Image', 'True Mask', 'Predicted Mask']

        for img, mask in dataset.take(num):
            plt.figure(figsize=(15, 15))
            pred_mask = self.model.predict(img)
            images = [img[0], mask[0], pred_mask[0]]

            for i in range(len(images)):
                plt.subplot(1, len(images), i + 1)
                plt.title(titles[i])
                plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i]))
                plt.axis('off')
            plt.show()

    def plot_history(self):
        """
        Plots the Dice coefficient and accuracy per epoch.
        """
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        dice = self.history.history['dice_coef']
        val_dice = self.history.history['val_dice_coef']

        plt.figure()
        plt.plot(self.history.epoch, acc, 'r', label='Training Accuracy')
        plt.plot(self.history.epoch, val_acc, 'bo', label='Validation loss')
        plt.plot(self.history.epoch, dice, 'gold', label="Training dice")
        plt.plot(self.history.epoch, val_dice, 'green', label="Val Dice")
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
