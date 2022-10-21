import tensorflow as tf
import matplotlib.pyplot as plt

from utils import init_model

def plot_loss(history, p):
      acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']

      loss = history.history['loss']
      val_loss = history.history['val_loss']

      epochs_range = range(p.epochs())

      plt.figure(figsize=(8, 8))
      plt.subplot(1, 2, 1)
      plt.plot(epochs_range, acc, label='Training Accuracy')
      plt.plot(epochs_range, val_acc, label='Validation Accuracy')
      plt.legend(loc='lower right')
      plt.title('Training and Validation Accuracy')

      plt.subplot(1, 2, 2)
      plt.plot(epochs_range, loss, label='Training Loss')
      plt.plot(epochs_range, val_loss, label='Validation Loss')
      plt.legend(loc='upper right')
      plt.title('Training and Validation Loss')
      plt.savefig(p.image_dir() + "loss_acc.png")
      plt.show()

if __name__ == "__main__":
      train_ds, test_ds, valid_ds, preprocessing, model, p = init_model()

      reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, min_lr=1e-5, verbose=1)

      history = model.fit(x=train_ds,
            epochs=p.epochs(),
            validation_data=valid_ds, callbacks=[reduce_lr])

      plot_loss(history, p)

      model.save_weights(p.data_dir() + "checkpoints/my_checkpoint")