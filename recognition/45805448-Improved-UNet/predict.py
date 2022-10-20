import matplotlib.pyplot as plt
from keras.models import load_model
from dataset import load_dataset
from train import Trainer

class Predictor:
    def __init__(self):
        self.image_size = None

        self.trainer = None

        self.test_dataset = None

        self.model = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def load_test_data_and_model(self):
        self.test_dataset = self.trainer.test_dataset
        self.trainer.load_model()
        self.model = self.trainer.model

    def evaluate_model(self):
        self.model.evaluate(self.trainer.test_dataset, verbose=2)

    def display_samples_from_model(self):

        plt.figure(figsize=(10,10))
        for batch in self.trainer.test_dataset.take(1):
            test_images, test_masks = batch[0], batch[1]
            predicted_masks = self.model.predict(test_images)
            for i in range(self.trainer.batch_size):

                plt.subplot(5, 3, i*3 + 1)
                plt.imshow(test_images[i])
                plt.axis('off')
                plt.title('Image')

                plt.subplot(5, 3, i*3+ 2)
                plt.imshow(predicted_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Predicted Mask')

                plt.subplot(5, 3, i*3 + 3)
                plt.imshow(test_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Actual Mask')

        plt.savefig('predicted_samples.png')