from modules import create_model
from dataset import DataSet


class TrainedModel:

    def __int__(self):
        self.dataset = DataSet()
        self.model = create_model(self.dataset.image_shape)

