import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from perceiver import train, Perceiver
import random, os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

IMAGE_DIR = './AKOA_Analysis'
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

def generate_samples(instanceIds, dataset_proportion):

    samples = dict()
    trainIndex = []
    testIndex = []
    for fileId in instanceIds:
        fileName = fileId.split('_')[0]
        if fileName in samples:
            samples[fileName].append(fileId)
        else:
            samples[fileName] = [fileId]
    print("unique patients in entire dataset: ", len(samples))
    building_train = True
    for imgIndex in samples.values():
        for fileId in imgIndex:
            if building_train:  # first step: building training set
                if dataset_proportion * len(instanceIds) < len(trainIndex):
                    building_train = False  # start building test set now
                    break
                else:
                    trainIndex.append(fileId)
            else:  # second step: building testing set
                if len(testIndex) <= len(instanceIds) * (1-dataset_proportion):
                    testIndex.append(fileId)
                else:
                    break  # done building test set
    return trainIndex, testIndex

def loadInput(dir_data, dataset_proportion):

    all_instanceId = os.listdir(dir_data)

    trainIndex, testIndex = generate_samples(all_instanceId,
                                                            dataset_proportion)
    random.shuffle(trainIndex)
    random.shuffle(testIndex)

    img_shape = (64, 64)
    def load_samples(instanceIds):
        """
        Helper function for loading a X and y set based of the image names
        Args:
            instanceIds: The image names to build the data set from
        Returns:
            X_set, y_set: the tf array of the X and y set built
        """
        X_set = []
        y_set = []
        for i, fileId in enumerate(instanceIds):
            image = load_img(dir_data + "/" + fileId,
                             target_size=(img_shape), color_mode="grayscale")

            image = img_to_array(image)
            X_set.append(image)
            if "LEFT" in fileId or "L_E_F_T" in fileId or \
                    "Left" in fileId or "left" in fileId:
                label = 0
            else:
                label = 1
            y_set.append(label)
        X_set = np.array(X_set)
        X_set /= 255.0
        return X_set, np.array(y_set, dtype=np.uint8).flatten()
    X_train, y_train = load_samples(trainIndex)
    X_val, y_val = load_samples(testIndex)
    X_test, y_test = X_val[len(X_val) // 5 * 4:], y_val[len(y_val) // 5 * 4:]
    X_val, y_val = X_val[0:len(X_val) // 5 * 4], y_val[0:len(y_val) // 5 * 4]
    return X_train, y_train, X_val, y_val, X_test, y_test
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

def process(image,label):
    image = tf.cast(image / 255. ,tf.float32)
    return image,label

def plot_data(history):
    # Plotting the Learning curves
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

    dataset_proportion = 0.8
    X_train, y_train, X_val, y_val, X_test, y_test = loadInput(IMAGE_DIR, dataset_proportion)
    print(len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(y_test))
    print(X_train.shape)

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




    # checkpoint.restore(ckpt_manager.latest_checkpoint)
    history = train(knee_model,
                    train_set=(X_train, y_train),
                    val_set=(X_val, y_val),
                    test_set=(X_test, y_test),
                    batch_size=BATCH_SIZE)


    plot_data(history)

    # Retrieve a batch of images from the test set
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