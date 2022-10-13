from modules import siamese
from modules import classification_model
from modules import contrastive_loss
from dataset import load_siamese_data
from dataset import load_classify_data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

SNN_PATH = 'recognition\\45853047-SiameseADNI\\models\\SNN.h5'
CLASSIFIER_PATH = 'recognition\\45853047-SiameseADNI\\models\\Classifier.h5'


def train():
    # Train and Save the SNN
    siamese_fit = trainSNN()

    # Train and Save classification
    classifier_fit = trainClassifier()

    # Plot Accuracy, Val Accuracy and Loss-----------
    plot_data(siamese_fit, [0, 50])
    plot_data(classifier_fit, [0, 1])
    plt.show()


def trainSNN():
    # Siamese model data
    siamese_train, siamese_val = load_siamese_data()

    # Siamese model
    model = siamese(128, 128)

    # Train
    siamese_fit = model.fit(siamese_train, epochs=30, validation_data=siamese_val)
    model.save(SNN_PATH)

    return siamese_fit


def trainClassifier():
    # Classification model data
    classify_train, classify_val = load_classify_data(testing=False)

    # Get the trained subnet
    siamese_model = load_model(SNN_PATH, custom_objects={ 'contrastive_loss': contrastive_loss })
    classifier = classification_model(siamese_model.get_layer(name="subnet"))

    # train
    classifier_fit = classifier.fit(classify_train, epochs=10, validation_data=classify_val)
    classifier.save(CLASSIFIER_PATH)

    return classifier_fit


def plot_data(fit, lim):
    plt.figure()
    plt.plot(fit.history['accuracy'], label='accuracy')
    plt.plot(fit.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot(fit.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(lim)
    plt.legend(loc='lower right')

train()