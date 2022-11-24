import os
import matplotlib.pyplot as plt
import tensorflow as tf
from train import train_isic_dataset

class Predictor:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate_model(self):
        """
        Evaluates the model based on the testing dataset split from training.
        """
        loss, dice_coef = self.trainer.model.evaluate(self.trainer.test_dataset, verbose=2)
        print(f'Model evaluation on test dataset: loss = {loss}, dice_coef = {dice_coef}')

    def output_predictions(self):
        """
        Predicts a random batch from the test dataset, and recovers the images. Saves figures of the
        input image, predicted mask, and actual mask into storage.
        """
        if not os.path.isdir(self.trainer.plots_path):
            os.makedirs(self.trainer.plots_path)

        plt.figure(figsize=(8,8))
        for batch in self.trainer.test_dataset.shuffle(buffer_size=10).take(1):
            test_images, test_masks = batch[0], batch[1]
            predicted_masks = self.trainer.model.predict(test_images)
            # Convert one-hot encoded masks to rgb channels
            colors = tf.Variable([[90, 0, 90], [255, 255, 0]]) # purple, yellow
            test_masks = tf.gather_nd(colors, tf.expand_dims(tf.cast(tf.argmax(test_masks, axis=-1), dtype=tf.int32), axis=-1))
            predicted_masks = tf.gather_nd(colors, tf.expand_dims(tf.cast(tf.argmax(predicted_masks, axis=-1), dtype=tf.int32), axis=-1))
            
            for i in range(self.trainer.batch_size):
                plt.subplot(self.trainer.batch_size, 3, i*3 + 1)
                plt.imshow(test_images[i])
                plt.axis('off')
                tf.keras.utils.save_img(self.trainer.plots_path + f'/ti{i}.png', test_images[i])

                plt.subplot(self.trainer.batch_size, 3, i*3+ 2)
                plt.imshow(predicted_masks[i])
                plt.axis('off')
                tf.keras.utils.save_img(self.trainer.plots_path + f'/pm{i}.png', predicted_masks[i])

                plt.subplot(self.trainer.batch_size, 3, i*3 + 3)
                plt.imshow(test_masks[i])
                plt.axis('off')
                tf.keras.utils.save_img(self.trainer.plots_path + f'/am{i}.png', test_masks[i])

        plt.savefig(self.trainer.plots_path + '/predicted_samples.png')
        print(f'Saved prediction images and plot to {self.trainer.plots_path}')

def predict_isic_dataset(trainer=None, images_path='', masks_path='', dataset_path='', model_path='', plots_path=''):
    """
    Main driver for predicting test data on the trained Improved UNet model using the ISIC dataset.
    """
    if trainer is None:
        trainer = train_isic_dataset(images_path, masks_path, dataset_path, model_path, plots_path)

    predictor = Predictor(trainer)
    predictor.evaluate_model()

    predictor.output_predictions()

    return predictor
