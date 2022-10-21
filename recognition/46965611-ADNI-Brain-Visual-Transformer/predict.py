"""
predict.py

Script for showing example usage of trained model.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from dataset import load_data
from modules import EmbedPatch, MultiHeadAttentionLSA, PatchLayer
from parameters import LEARNING_RATE, MODEL_SAVE_PATH, WEIGHT_DECAY


def predict(load_path, test_data):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    model = tf.keras.models.load_model(
        load_path,
        custom_objects={
            'PatchLayer': PatchLayer,
            'EmbedPatch': EmbedPatch,
            'MultiHeadAttentionLSA': MultiHeadAttentionLSA,
            'AdamW': optimizer
        }
    )
    
    model.evaluate(test_data)

    # Plot confusion matrix
    y_true = []
    y_pred = []

    for image_batch, label_batch in test_data:
        y_true.append(label_batch)
        y_pred.append((model.predict(image_batch, verbose=0) > 0.5).astype('int32'))

    labels_true = tf.concat([tf.cast(item[0], tf.int32) for item in y_true], axis=0)
    labels_pred = tf.concat([item[0] for item in y_pred], axis=0)

    matrix = tf.math.confusion_matrix(labels_true, labels_pred, 2).numpy()

    fig, ax = plt.subplots(figsize=(8,8))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual Label', fontsize=18)
    plt.suptitle('Confusion Matrix', fontsize=18)
    plt.savefig('confusion_matrix')
    plt.clf()


if __name__ == '__main__':
    train, val, test = load_data()
    predict(MODEL_SAVE_PATH, test)