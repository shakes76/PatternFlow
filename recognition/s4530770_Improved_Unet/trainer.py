from improved_unet import build_model
from keras import backend as K
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import DataLoader
print(keras.__version__)

# https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c

model = build_model((384, 512, 3))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', dice_coef])


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(dataset, num=1):
    for img, mask in dataset.take(num):
        pred_mask = model.predict(img)
        display([img[0], mask[0], create_mask(pred_mask)])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(data)
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

EPOCHS = 5
data = DataLoader("H:\\COMP3710\\ISIC2018_Task1-2_Training_Data\\", batch_size=16)
#show_predictions(data.get_training_set())
history = model.fit(data.get_training_set(),
                    epochs=EPOCHS,
                    validation_data=data.get_validation_set(),
                    callbacks=[DisplayCallback()])

loss = history.history['loss']
val_loss = history.history['val_loss']
dice_coef = history.history['dice_coef']
val_dice_coef = history.history['val_dice_coef']

plt.figure()
plt.plot(history.epoch, loss, 'r', label='Training loss')
plt.plot(history.epoch, val_loss, 'bo', label='Validation loss')
plt.plot(history.epoch, dice_coef, 'gold', label="Dice Coefficients")
plt.plot(history.epoch, val_dice_coef, 'green', label="Dice Coefficients")
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()