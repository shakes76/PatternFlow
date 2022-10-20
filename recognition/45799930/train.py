from modules import create_model
from dataset import DataSet
from tensorflow import reduce_sum
import matplotlib.pyplot as plt


def train_model():
    dataset = DataSet()
    model = create_model()

    # Train the model
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[dice_sim_co])
    history = model.fit(dataset.training.batch(10), epochs=10, validation_data=dataset.validate.batch(10))
    print_history(history)

    # Print the model history


def print_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def dice_sim_co(x, y):
    """
    This implements the dice similarity coefficients as described on the link in the task sheet.
    To get the coefficient the equation is
    (2 * |X ∩ Y|)/(|X| + |Y|)
    :param x : An image in the form of a tensor
    :param y : An image to compare to in the form of a tensor.
    :return:
    """
    return (2 * reduce_sum(x * y)) / (reduce_sum(x) + reduce_sum(y))
