import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from module import improved_model


BATCH_SIZE = 32

def processing_jpg(path):
    """
        Image processing function
        choosing the image size of 192, 256
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (192, 256))
    
    return image
  
def processing_png(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, (192, 256))
    
    return image

def normal(image, ground):
    """
        normalize function
    """
    image = tf.cast(image, tf.float32) / 255.0
    ground = tf.cast(ground, tf.float32) / 255.0
    
    return image, ground

def load_image(image_path, ground_path):
    image = processing_jpg(image_path)
    ground = processing_png(ground_path)
    image, ground = normal(image, ground)

    return image, ground

def data_processing(image_input, ground_truth):
    # Shuffle pictures
    index = np.random.permutation(len(image_input))
    image_input = np.array(image_input)[index]
    ground_truth = np.array(ground_truth)[index]
    
    # Divide the dataset into training set, validation set and test set with 7：2：1
    length = len(image_input)
    image_input_val = image_input[:(int(length*0.2))]
    ground_truth_val = ground_truth[:(int(length*0.2))]

    image_input_test = image_input[int(length*0.2):int(length*0.3)]
    ground_truth_test = ground_truth[int(length*0.2):int(length*0.3)]

    image_input_train = image_input[int(length*0.3):]
    ground_truth_train = ground_truth[int(length*0.3):]
    
    train_ds = tf.data.Dataset.from_tensor_slices((image_input_train, ground_truth_train))
    val_ds = tf.data.Dataset.from_tensor_slices((image_input_val, ground_truth_val))
    test_ds = tf.data.Dataset.from_tensor_slices((image_input_test, ground_truth_test))
    
    train_ds = train_ds.map(load_image)
    val_ds = val_ds.map(load_image)
    test_ds = test_ds.map(load_image)
    
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)
    
    return train_ds, val_ds, test_ds

def dice_coef(y_true, y_pred, smooth = 1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice_coef_result = (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    return dice_coef_result

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def plot_dice_and_loss(history):
    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(20)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dice, label='Training Dice')
    plt.plot(epochs_range, val_dice, label='Validation Dice')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Dice Coefficient')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_prediction_images(model, test_ds, n = 3):
    X_test, y_test = next(iter(test_ds))
    y_pred = model.predict(X_test)
    
    FONT_SIZE = 18
    
    plt.figure(figsize=(14, 14))
    for i in range(n):
        plt.subplot(n, 3, i*3+1)
        plt.imshow(X_test[i])
        plt.axis('off')
        plt.title("Image", size=FONT_SIZE)

        plt.subplot(n, 3, i*3+2)

        plt.imshow(y_pred[i])
        plt.gray()
        plt.axis('off')
        plt.title("Prediction", size=FONT_SIZE)

        plt.subplot(n, 3, i*3+3)
        plt.imshow(y_test[i])
        plt.gray()
        plt.axis('off')
        plt.title("Original_Result", size=FONT_SIZE)
    plt.show()

def main():
    image_input = sorted(tf.io.gfile.glob('./ISIC2018_Task1-2_Training_Input_x2/*.jpg'))
    ground_truth = sorted(tf.io.gfile.glob('./ISIC2018_Task1_Training_GroundTruth_x2/*.png'))
    
    train_ds, val_ds, test_ds = data_processing(image_input, ground_truth)
    
    model = improved_model()
    model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
    
    history = model.fit(train_ds, validation_data = val_ds, epochs = 20)
    
    plot_dice_and_loss(history)
    
    plot_prediction_images(model, test_ds)
    
    pred_result = model.evaluate(test_ds, verbose=1)
    print(pred_result)

if __name__ == "__main__":
    main()

