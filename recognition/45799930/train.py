from modules import create_model
from dataset import DataSet
from tensorflow import reduce_sum


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


class TrainedModel:

    def __int__(self):
        self.dataset = DataSet()
        self.model = create_model(self.dataset.image_shape)

