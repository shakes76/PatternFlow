import os
import matplotlib.pyplot as plt
import tensorflow as tf
from train import train_isic_dataset

class Predictor:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate_model(self):
        loss, dice_coef = self.trainer.model.evaluate(self.trainer.test_dataset, verbose=2)
        print(f'Model evaluation on test dataset: loss = {loss}, dice_coef = {dice_coef}')

    def output_predictions(self):
        if not os.path.isdir(self.trainer.plots_path):
            os.makedirs(self.trainer.plots_path)

        plt.figure(figsize=(8,8))
        for batch in self.trainer.test_dataset.shuffle(buffer_size=10).take(1):
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

        plt.savefig(self.trainer.plots_path + '/predicted_samples.png')
        print(f'Saved plot to {self.trainer.plots_path}/predicted_samples.png')

def predict_isic_dataset(trainer=None, images_path='', masks_path='', dataset_path='', model_path='', plots_path=''):
    if trainer is None:
        trainer = train_isic_dataset(images_path, masks_path, dataset_path, model_path, plots_path)

    predictor = Predictor(trainer)
    predictor.evaluate_model()

    predictor.output_predictions()
