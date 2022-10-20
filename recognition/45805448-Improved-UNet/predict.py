import matplotlib.pyplot as plt
from keras.models import load_model
from dataset import load_dataset

class Predictor:
    def __init__(self):
        self.image_size = None

        self.datasets_path = None
        self.model_path = None

        self.test_dataset = None

        self.model = None

    def set_paths(self, datasets_path, model_path):
        self.datasets_path = datasets_path
        self.model_path = model_path

    def load_test_data_and_model(self):
        self.test_dataset = load_dataset(self.datasets_path + '/test')
        self.model = load_model(self.model_path)

    def evaluate_model(self):
        self.model.evaluate(self.test_dataset, verbose=2)

    def display_samples_from_model(self):
        test_images, test_masks = self.test_dataset.take(1)
        predicted_masks = self.model.predict(test_images)

        plt.figure(figsize=(10,10))
        for i in range(3):
            plt.subplot(3, 3, i*3 + 1)
            plt.imshow(test_images[i])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(3, 3, i*3 + 2)
            plt.imshow(predicted_masks[i], vmin=0, vmax=1)
            plt.axis('off')
            plt.title('Predicted Mask')

            plt.subplot(3, 3, i*3 + 3)
            plt.imshow(test_masks[i], vmin=0, vmax=1)
            plt.axis('off')
            plt.title('Actual Mask')

        plt.savefig('predicted_samples.png')