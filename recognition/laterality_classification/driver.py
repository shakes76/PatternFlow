from recognition.laterality_classification.laterality_classifier import *
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import random


def proof_no_set_overlap(train_image_names, test_image_names):
    unique_train_patients = []
    unique_test_patients = []

    for train_name in train_image_names:
        if train_name[:10] not in unique_train_patients:
            unique_train_patients.append(train_name[:10])
    for test_name in test_image_names:
        if test_name[:10] not in unique_test_patients:
            unique_test_patients.append(test_name[:10])

    print("unique patients in training set: ", len(unique_train_patients))
    print("unique patients in testing set: ", len(unique_test_patients))

    matches = len([x for x in unique_train_patients
                   if x in unique_test_patients])

    print("number of patients in training and testing: ", matches)


def split_by_patients(image_names, N_train, N_test):
    patient_batches = dict()
    train_image_names = []
    test_image_names = []

    for name in image_names:
        patient_id = name.split('_')[0]
        if patient_id in patient_batches:
            patient_batches[patient_id].append(name)
        else:
            patient_batches[patient_id] = [name]
    print("unique patients in entire dataset: ", len(patient_batches))

    building_train = True
    for patient_batch in patient_batches.values():
        for name in patient_batch:
            if building_train:  # first step: building training set
                if len(train_image_names) < N_train:
                    train_image_names.append(name)
                else:
                    building_train = False  # start building test set now
                    break
            else:  # second step: building testing set
                if len(test_image_names) < N_test:
                    test_image_names.append(name)
                else:
                    break  # done building test set

        if len(test_image_names) >= N_test:
            break

    return train_image_names, test_image_names


def process_dataset(dir_data, N_train, N_test):
    all_image_names = os.listdir(data_dir)
    # num_total: 18680
    # num left found: 7,760
    # num_right found: 10,920

    random.shuffle(all_image_names)

    train_image_names, test_image_names = split_by_patients(all_image_names,
                                                            N_train, N_test)

    random.shuffle(train_image_names)
    random.shuffle(test_image_names)

    proof_no_set_overlap(train_image_names, test_image_names)

    img_shape = (228, 260, 3)

    def get_data(image_names):
        X_set = []
        y_set = []
        for i, name in enumerate(image_names):
            image = load_img(dir_data + "/" + name,
                             target_size=img_shape[:2])

            # normalise image pixels
            image = img_to_array(image)

            X_set.append(image[:, :, 0])
            if "LEFT" in name or "L_E_F_T" in name or \
                    "Left" in name or "left" in name:
                label = 0
            else:
                label = 1

            y_set.append(label)

        X_set = tf.convert_to_tensor(np.array(X_set, dtype=np.uint8))
        X_set = tf.cast(X_set, tf.float16) / 255.0

        return X_set, tf.convert_to_tensor(y_set)

    X_train, y_train = get_data(train_image_names)
    X_test, y_test = get_data(test_image_names)

    return X_train, y_train, X_test, y_test


def visualise_images(X, y, set_size, title, seed=0):
    random.seed(seed)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        index = random.randint(0, set_size - 1)
        plt.subplot(3, 3, i + 1)
        # cast back to float32 for visualisation
        plt.imshow(tf.cast(X[index], tf.float32), cmap='gray')
        label = y[index]
        plt.title("left" if label == 0 else "right")
        plt.axis('off')

    plt.suptitle(title)
    plt.show()


def mean_square_error(predictions, actual):
    return sum([(y - yhat) ** 2 for (y, yhat) in (actual, predictions)]) \
           / len(actual)


def build_train_model(classifier, X_train, y_train, X_test, y_test):
    model = classifier.build_simple_model()

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    # format input to 4 dims
    X_train = X_train[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]

    model_history = model.fit(x=X_train, y=y_train,
                              validation_data=(X_test, y_test),
                              epochs=3)
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('accurary over epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['training', 'validation'])
    plt.show()

    test_predictions = tf.cast(tf.round(model.predict(X_test)), tf.int32)[:, 0]

    acc = len([i for i in range(len(y_test))
               if test_predictions[i] == y_test[i]]) / len(y_test)
    print(f"validation acc = {acc}")

    visualise_images(X_test[:, :, :, 0], test_predictions,
                     100, "Model predictions on validation set", seed=69)

    visualise_images(X_test[:, :, :, 0], y_test,
                     100, "Actual labels of validation set", seed=69)


if __name__ == "__main__":
    # data_dir = "H:/Desktop/AKOA_Analysis/"

    # load up dataset, download if necessary
    url = "https://cloudstor.aarnet.edu.au/sender/download.php?token=d82346d9-f3ca-48bf-825f-327a622bfaca&files_ids=9881639"
    data_dir = tf.keras.utils.get_file(origin=url,
                                       fname='AKOA_Analysis',
                                       untar=True)

    train_size = 6000
    test_size = 1500

    # split up dataset, 12000 images for training, 3000 for testing
    X_train, y_train, X_test, y_test = process_dataset(data_dir, train_size, test_size)

    print("proportion of right knee in training set:",
          len([x for x in y_train if x == 1]), len(y_train))
    print("proportion of right knee in test set:",
          len([x for x in y_test if x == 1]), len(y_test))

    # visualise some of training set
    visualise_images(X_train, y_train, train_size, "visualisation of training set")
    visualise_images(X_test, y_test, test_size, "visualisation of testing set")

    classifier = LateralityClassifier((228, 260, 1))

    build_train_model(classifier, X_train, y_train, X_test, y_test)
