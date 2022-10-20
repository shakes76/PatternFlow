import os
import matplotlib.pyplot as plt
import tensorflow as tf
from train import train_isic_dataset, PLOT_SAMPLES_PATH

PLOT_PREDICTIONS_PATH = 'predicted_samples.png'

class Predictor:
    def __init__(self, trainer, plot_predictions_path):
        self.plot_predictions_path = plot_predictions_path

        self.trainer = trainer

    def evaluate_model(self):
        self.trainer.model.evaluate(self.trainer.test_dataset, verbose=2)

    def output_predictions(self):
        plt.figure(figsize=(8,8))
        for batch in self.trainer.test_dataset.take(1):
            test_images, test_masks = batch[0], tf.argmax(batch[1], axis=-1)
            predicted_masks = tf.argmax(self.trainer.model.predict(test_images), axis=-1)
            for i in range(self.trainer.batch_size):

                plt.subplot(self.trainer.batch_size, 3, i*3 + 1)
                plt.imshow(test_images[i])
                plt.axis('off')
                plt.title('Image')

                plt.subplot(self.trainer.batch_size, 3, i*3+ 2)
                plt.imshow(predicted_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Predicted Mask')

                plt.subplot(self.trainer.batch_size, 3, i*3 + 3)
                plt.imshow(test_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Actual Mask')

        plt.savefig(self.plot_predictions_path)

def predict_isic_dataset(trainer=None, images_path='', masks_path='', dataset_path='', model_path='', 
                            plot_samples_path=PLOT_SAMPLES_PATH, plot_predictions_path=PLOT_PREDICTIONS_PATH,
                            override_predictions=False):
    if trainer is None:
        trainer = train_isic_dataset(images_path, masks_path, dataset_path, model_path, plot_samples_path)

    predictor = Predictor(trainer, plot_predictions_path)
    predictor.evaluate_model()

    if override_predictions or not os.path.exists(predictor.plot_predictions_path):
        predictor.output_predictions()
