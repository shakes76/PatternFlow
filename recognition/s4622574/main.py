import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from perceiver import fitModel, Perceiver
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
                if (1-dataset_proportion) * len(instanceIds) >= len(testIndex):
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
            input, list_of_labels: the tf array of the X and y set built
        """
        input = []
        list_of_labels = []
        for i, fileId in enumerate(instanceIds):
            image = load_img(dir_data + "/" + fileId,
                             target_size=(img_shape), color_mode="grayscale")

            image = img_to_array(image)
            input.append(image)
            if "RIGHT" in fileId or "R_I_G_H_T" in fileId or "Right" in fileId or "right" in fileId:
                label = 1
            else:
                label = 0
            list_of_labels.append(label)
        input = np.array(input)
        input /= 255.0
        return input, np.array(list_of_labels, dtype=np.uint8).flatten()
    trainX, trainY = load_samples(trainIndex)
    valX, valY = load_samples(testIndex)
    testX, testY = valX[len(valX) // 5 * 4:], valY[len(valY) // 5 * 4:]
    valX, valY = valX[0:len(valX) // 5 * 4], valY[0:len(valY) // 5 * 4]
    return trainX, trainY, valX, valY, testX, testY
def create_dataset(image_dir, img_size):

    split_train = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                    subset="training", seed=46, label_mode='int',
                                                    batch_size=1, image_size=img_size)

    split_val = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                      subset="validation", seed=46, label_mode='int',
                                                      batch_size=1, image_size=img_size)

    foldSize = tf.data.experimental.cardinality(split_val)
    split_test = split_val.take(foldSize // 5)
    split_val = split_val.skip(foldSize // 5)

    split_train = split_train.map(process).prefetch(AUTO_TUNE)
    split_val = split_val.map(process).prefetch(AUTO_TUNE)
    split_test = split_test.map(process).prefetch(AUTO_TUNE)
    return split_train, split_val, split_test

def process(inputImg, groundTruth):
    inputImg = tf.cast(inputImg / 255. ,tf.float32)
    return inputImg, groundTruth

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
    trainX, trainY, valX, valY, testX, testY = loadInput(IMAGE_DIR, dataset_proportion)
    print(len(trainX), len(trainY), len(valX), len(valY), len(testX), len(testY))
    print(trainX.shape)

    perceiverTransformer = Perceiver(patch_size=0,
                            data_size=ROWS*COLS, 
                            latent_size=LATENT_SIZE,
                            freq_ban=NUM_BANDS,
                            proj_size=PROJ_SIZE, 
                            num_heads=NUM_HEADS,
                            num_trans_blocks=NUM_TRANS_BLOCKS,
                            num_iterations=NUM_ITER,
                            max_freq=MAX_FREQ,
                            lr=LR,
                            weight_decay=WEIGHT_DECAY,
                            epoch=EPOCHS)

 


    # checkpoint.restore(ckpt_manager.latest_checkpoint)
    history = fitModel(perceiverTransformer,
                    train_set=(trainX, trainY),
                    val_set=(valX, valY),
                    test_set=(testX, testY),
                    batch_size=BATCH_SIZE)


    plot_data(history)
    print("Evaluation")
    # Retrieve a batch of images from the test set
    testData, testClass = testX[:BATCH_SIZE], testY[:BATCH_SIZE]
    testData = testData.reshape((BATCH_SIZE, ROWS, COLS, 1))
    evaluation = perceiverTransformer.predict_on_batch(testData).flatten()
    testClass = testClass.flatten()

    evaluation = tf.where(evaluation < 0.5, 0, 1).numpy()
    label = {0: "left", 1: "right"}

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(testData[i], cmap="gray")
        plt.title("pred: " + label[evaluation[i]] + ", real: " + label[testClass[i]])
        plt.axis("off")
    plt.show()
    print("Finished")