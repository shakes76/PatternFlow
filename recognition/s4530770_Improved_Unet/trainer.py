from improved_unet import build_model
from keras import backend as K
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import DataLoader

# https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c




model = build_model((384, 512, 3), 32)
#model = get_model((384, 512), 1)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss=dice_coef_loss,
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


EPOCHS = 5
data = DataLoader("C:\\Users\\s4530770\\Downloads\\ISIC2018_Task1-2_Training_Data\\", batch_size=2)
#show_predictions(data.get_training_set())
train_ds = data.get_training_set()
val_ds = data.get_validation_set()
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds)

show_predictions(data.get_test_set(), 3)
test_loss, test_acc = model.evaluate(data.get_test_set(), verbose=1)
print(test_acc, test_loss)
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