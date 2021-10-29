import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from model import train, Perceiver

IMAGE_DIR = 'D:/AKOA_Analysis'
BATCH_SIZE = 32
IMG_SIZE = (64, 64) # image resize
ROWS, COLS = IMG_SIZE
TEST_PORTION = 5 # portion of validation set to become test set
SHUFFLE_RATE = 512
AUTO_TUNE = tf.data.experimental.AUTOTUNE

LATENT_SIZE = 256  # Size of the latent array.
NUM_BANDS = 4 # Number of bands in Fourier encode. Used in the paper
NUM_CLASS = 1 # Number of classes to be predicted (1 for binary)
PROJ_SIZE = 2*(2*NUM_BANDS + 1) + 1  # Projection size of data after fourier encoding
NUM_HEADS = 8  # Number of Transformer heads.
NUM_TRANS_BLOCKS = 6 # Number of transformer blocks in the transformer layer. Used in the paper
NUM_ITER = 8  # Repetitions of the cross-attention and Transformer modules. Used in the paper
MAX_FREQ = 10 # Max frequency in Fourier encode
LR = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 10

def create_dataset(image_dir, img_size):

    training_dataset = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                    subset="training", seed=46, label_mode='int',
                                                    batch_size=1, image_size=img_size)

    validation_dataset = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                      subset="validation", seed=46, label_mode='int',
                                                      batch_size=1, image_size=img_size)


    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // TEST_PORTION)
    validation_dataset = validation_dataset.skip(val_batches // TEST_PORTION)


    training_dataset = training_dataset.map(process).prefetch(AUTO_TUNE)
    validation_dataset = validation_dataset.map(process).prefetch(AUTO_TUNE)
    test_dataset = test_dataset.map(process).prefetch(AUTO_TUNE)

    return training_dataset, validation_dataset, test_dataset





def dataset_to_numpy_util(dataset, len_ds):
  dataset = dataset.batch(len_ds)
  for images, labels in dataset:
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    break

  return numpy_images, numpy_labels

"""
Normalize image to range [0,1]
"""
def process(image,label):
    image = tf.cast(image / 255. ,tf.float32)
    return image,label

def get_numpy_ds():
    training_set, validation_set, test_set = create_dataset(IMAGE_DIR, IMG_SIZE)
    X_train, y_train = dataset_to_numpy_util(training_set, len(training_set))
    X_train = X_train.reshape((len(training_set), ROWS, COLS, 1))

    X_val, y_val = dataset_to_numpy_util(validation_set, len(validation_set))
    X_val = X_val.reshape((len(validation_set), ROWS, COLS, 1))

    X_test, y_test = dataset_to_numpy_util(test_set, len(test_set))
    X_test = X_test.reshape((len(test_set), ROWS, COLS, 1))
    del training_set
    del validation_set
    del test_set
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def plot_data(history):

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

if __name__ == "__main__":


    X_train, y_train, X_val, y_val, X_test, y_test = get_numpy_ds()


    knee_model = Perceiver(patch_size=0,
                            data_size=ROWS*COLS, 
                            latent_size=LATENT_SIZE,
                            num_bands=NUM_BANDS,
                            proj_size=PROJ_SIZE, 
                            num_heads=NUM_HEADS,
                            num_trans_blocks=NUM_TRANS_BLOCKS,
                            num_iterations=NUM_ITER,
                            max_freq=MAX_FREQ,
                            lr=LR,
                            weight_decay=WEIGHT_DECAY,
                            epoch=EPOCHS)



    history = train(knee_model,
                    train_set=(X_train, y_train),
                    val_set=(X_val, y_val),
                    test_set=(X_test, y_test),
                    batch_size=BATCH_SIZE)


    plot_data(history)


    image_batch, label_batch = X_test[:BATCH_SIZE], y_test[:BATCH_SIZE]
    image_batch = image_batch.reshape((BATCH_SIZE, ROWS, COLS, 1))
    predictions = knee_model.predict_on_batch(image_batch).flatten()
    label_batch = label_batch.flatten()

    predictions = tf.where(predictions < 0.5, 0, 1).numpy()
    class_names = {0: "left", 1: "right"}

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i], cmap="gray")
        plt.title("pred: " + class_names[predictions[i]] + ", real: " + class_names[label_batch[i]])
        plt.axis("off")
    plt.show()