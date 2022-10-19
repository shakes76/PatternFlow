# This file contains the source code of the components of my model as functions or classes
import os
import sys
sys.path.insert(1, os.getcwd())
from dataset import loadFile

import tensorflow as tf
from tensorflow import data
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

def generatePairs(ad, nc, batch=8):
    print('>> Begin pair generation')
    # DataGenerator for weak augmentation
    datagen = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.15),
                                layers.experimental.preprocessing.RandomHeight(0.15),
                                layers.experimental.preprocessing.RandomWidth(0.2),
                                layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
                                layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.1)])
    print('> Zipping...')
    ad = ad.unbatch()
    nc = nc.unbatch()
    # Zipping the data into pairs and give them labels
    diff1 = (data.Dataset.zip((ad, nc))).map(lambda im1, im2: (im1, im2, 1.))
    diff2 = (data.Dataset.zip((nc, ad))).map(lambda im1, im2: (im1, im2, 1.))
    same1 = (data.Dataset.zip((ad, ad))).map(lambda im1, im2: (im1, im2, 0.))
    same2 = (data.Dataset.zip((nc, nc))).map(lambda im1, im2: (im1, im2, 0.))
    # Sample (concatinate) all four image-label pair datasets
    combined_ds = data.experimental.sample_from_datasets([diff1, diff2, same1, same2])
    combined_ds = combined_ds.batch(batch_size=batch)
    print("> Apply data augmentation... (It's ganna take a while)")
    # combined_ds.map(lambda im1, im2, l: (datagen(im1), datagen(im2), l)) # Wrong!!!!!
    for i1, i2, _ in combined_ds:
        i1 = datagen(i1)
        i2 = datagen(i2)
    print('>> Complete')
    return combined_ds
    

def makeCNN():
    # This CNN is almost the same as the one presented in the paper 
    input = layers.Input(shape=(224, 224, 1))
    conv = layers.Conv2D(64, 10, activation='relu', name='c0', padding='same')(input)
    pool = layers.MaxPooling2D(2)(conv)
    norm = layers.BatchNormalization()(pool)
    
    conv = layers.Conv2D(128, 7, activation='relu', name='c1', padding='same')(norm)
    pool = layers.MaxPooling2D(2)(conv)
    norm = layers.BatchNormalization()(pool)
    
    conv = layers.Conv2D(128, 4, activation='relu', name='c2', padding='same')(norm)
    pool = layers.MaxPooling2D(2)(conv)
    norm = layers.BatchNormalization()(pool)
    
    conv = layers.Conv2D(256, 4, activation='relu', name='c3', padding='same')(norm)
    norm = layers.BatchNormalization()(conv)
    
    flat = layers.Flatten(name='flat')(norm)
    # dense = layers.Dense(4096, activation='sigmoid')(flat)
    # dense = layers.Dense(1024, activation='sigmoid', name='d0', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(flat)
    out = layers.Dense(512, activation='sigmoid', name='out', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(flat)
    
    return Model(inputs=input, outputs=out, name='embeddingCNN')
    

def makeSiamese(cnn):
    EPS = 1e-8
    
    input_1 = layers.Input((224, 224, 1))
    input_2 = layers.Input((224, 224, 1))
    
    tower_1 = cnn(input_1)
    tower_2 = cnn(input_2)
    # Merging the two networks outputs (EPS is used to avoied 0 distance)
    distance = lambda v: tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(v[0] - v[1]), axis=1, keepdims=True), EPS))
    merge_layer = layers.Lambda(distance)([tower_1, tower_2])
    normal_layer = layers.BatchNormalization()(merge_layer)
    # 1 if same class, 0 if not
    output_layer = layers.Dense(1, activation="sigmoid", name='out2')(normal_layer)
    
    return Model(inputs=[input_1, input_2], outputs=output_layer, name='Siamese')


def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            (1 - y_true) * tf.math.square(y_pred) + y_true * tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        )
        
    def crossentropy(y_true, y_pred):
        tf.keras.losses.BinaryCrossentropy(y_true, y_pred)
    return contrastive_loss

def visualize(img1, img2, labels, to_show=6, num_col=3, predictions=None, test=False):


    num_row = to_show // num_col if to_show // num_col != 0 else 1

    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow((tf.concat([img1[i], img2[i]], axis=1).numpy()*255.0).astype("uint8"), cmap='gray', vmin=0, vmax=255)
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()
    
def main():
    # Code for testing the functions
    t_a, t_n, v_a, v_n = loadFile('F:/AI/COMP3710/data/AD_NC/')
    d = generatePairs(t_a, t_n)
    # visualize the data
    for img1, img2, label in d.take(1):
        visualize(img1, img2, label, to_show=8, num_col=2)
        break
        
    # cnn = makeCNN()
    # print(cnn.summary())
    # siamese = makeSiamese(cnn)
    # print(siamese.summary())

if __name__ == "__main__":
    main()
        
    
