import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np

def process_data(dir, train_count, test_count):
    """
    Process the AKOA dataset
    """
    all_images = os.listdir(dir)

    patients = {}

    train_images = []
    test_images = []

    for image in all_images:
        patient_name = image.split("_")[0]
        if patient_name in patients:
            patients[patient_name].append(image)
        else:
            patients[patient_name] = [image]
    

    dict_keys = list(patients.keys())
    for key in dict_keys[:train_count]:
        for image in patients[key]:
            train_images.append(image)
    
    for key in dict_keys[train_count: train_count + test_count]:
        for image in patients[key]:
            test_images.append(image)
        

    # randomise dataset
    random.shuffle(train_images)
    random.shuffle(test_images)
    
    image_size = (228, 260, 3)

    X_train, y_train = get_label(dir, train_images, image_size)
    X_test, y_test = get_label(dir, test_images, image_size)

    return X_train, y_train, X_test, y_test


def get_label(dir, images, image_size):

    processed_images = []
    labels = []

    for i, image_name in enumerate(images):
        image = load_img(dir + "/" + image_name, target_size = image_size)

        image = img_to_array(image)

        processed_images.append(image[:, :, 0])
    
        if "RIGHT" in image_name or "R_I_G_H_T" in image_name or "Right" in image_name or "right" in image_name:
            label = 1
        else:
            label = 0

        labels.append(label)

    processed_images = tf.convert_to_tensor(np.array(processed_images, dtype=np.uint8))
    processed_images = processed_images.cast(processed_images, tf.float16) / 255.0
    labels = tf.convert_to_tensor(labels)

    return processed_images, labels



def main():
    process_data("AKOA_Analysis\AKOA_Analysis", 80, 20)

if __name__ == "__main__":
    main()