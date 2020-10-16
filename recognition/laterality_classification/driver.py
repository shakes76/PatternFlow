from recognition.laterality_classification.laterality_classifier import *
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import random


def process_dataset(dir_data, N_train, N_test):
    all_image_names = os.listdir(data_dir)
    random.shuffle(all_image_names)
    # num_total: 18680
    # num left found: 7,760
    # num_right found: 10,920

    train_image_names = all_image_names[:N_train]

    test_image_names = all_image_names[N_train:N_train + N_test]
    img_shape = (228, 260, 3)

    def get_data(image_names):
        X_set = []
        y_set = []
        for i, name in enumerate(image_names):
            image = load_img(dir_data + "/" + name,
                             target_size=img_shape[:2])

            # normalise image pixels
            image = img_to_array(image) / 255.0

            X_set.append(image)
            if "LEFT" in name or "L_E_F_T" in name or \
                    "Left" in name or "left" in name:
                label = 0
            else:
                label = 1

            y_set.append(label)

        X_set = np.array(X_set)

        return tf.image.rgb_to_grayscale(tf.convert_to_tensor(X_set))[:, :, :, 0], \
               tf.convert_to_tensor(y_set)

    X_train, y_train = get_data(train_image_names)
    X_test, y_test = get_data(test_image_names)

    return X_train, y_train, X_test, y_test


def visualise_images(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        index = random.randint(0, 1000)
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[index], cmap='gray')
        label = y[index]
        plt.title("left" if label == 0 else "right")
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # data_dir = "H:/Desktop/AKOA_Analysis/"

    url = "https://cloudstor.aarnet.edu.au/sender/download.php?token=d82346d9-f3ca-48bf-825f-327a622bfaca&files_ids=9881639"
    data_dir = tf.keras.utils.get_file(origin=url,
                                       fname='AKOA_Analysis',
                                       untar=True)

    X_train, y_train, X_test, y_test = process_dataset(data_dir, 1000, 10)

    print(len([x for x in y_train if x == 1]), len(y_train))

    visualise_images(X_train, y_train)

    classifier = LateralityClassifier
