import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import pathlib
from PIL import Image
import zipfile
from model import improved_unet

def decode_png(file_path):
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png, channels=1)
    png = tf.image.resize(png, (256, 256))
    return png

def process_path(scan_fp, label_fp):
    #Process the MRI scan
    scan = decode_png(scan_fp)
    scan = tf.cast(scan, tf.float32) / 255.0

    #Process the Label image
    label = decode_png(label_fp)
    label = label == [0, 85, 170, 255]
    #tf.print(tf.unique(tf.reshape(label, [-1])))
    return scan, label

def display(display_list):
    plt.figure(figsize=(10,10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

def show_predictions(dataset, model):
    num = 1
    for scan, label in dataset.take(1):
        pred_label = model.predict(scan[tf.newaxis, ...])
        pred_label = tf.argmax(pred_label[0], axis=-1)
        display([tf.squeeze(scan), tf.argmax(label, axis=-1), pred_label])
    return (label, pred_label)

def dsc(label, pred_label, layer):
    print("Label: ")
    tf.print(tf.unique(tf.reshape(label, [-1])))
    print(label.shape)
    print("pred_label")
    tf.print(tf.unique(tf.reshape(pred_label, [-1])))
    print(pred_label[0][0])
    print(pred_label[0][0] == 0)
    print(pred_label[0][0] == 1)
    print(pred_label[0][0] == 2)
    print(pred_label[0][0] == 3)
    image_size = 256
    denominator = 0
    label_card = 0
    pred_label_card = 0
    for i in range(0, 256):
        for j in range(0, 256):
            if np.argmax(label[i][j]) == layer:
                if pred_label[i][j] == layer:
                    denominator += 1
                label_card += 1
            if pred_label[i][j] == layer:
                pred_label_card += 1
    numerator = label_card + pred_label_card
    return (2 * denominator) / (numerator)


def main():
    #Get the file location of where the data is stored.
    data_dir = pathlib.Path('H:/Year 3/Sem 2/COMP3710/Report/PatternFlow/recognition/keras_png_slices_data/keras_png_slices_data')
    #The following 5 lines were used to download and unzip the data, these have been commented out for faster debugging.
    #However, they have been left in to show how the files were retreived.
    #data_dir_tf = tf.keras.utils.get_file(origin='https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download',
    #                                      fname='H:/Year 3/Sem 2/COMP3710/Report/PatternFlow/recognition/keras_png_slices_data.zip')

    #with zipfile.ZipFile(data_dir_tf) as zf:
    #    zf.extractall()

    #List the file paths of the data
    test_scans = sorted(glob.glob('keras_png_slices_data/keras_png_slices_test/*.png'))
    test_labels = sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_test/*.png'))
    train_scans = sorted(glob.glob('keras_png_slices_data/keras_png_slices_train/*.png'))
    train_labels = sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_train/*.png'))
    val_scans = sorted(glob.glob('keras_png_slices_data/keras_png_slices_validate/*png'))
    val_labels = sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_validate/*.png'))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_scans, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_scans, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_scans, test_labels))

    train_dataset = train_dataset.shuffle(len(train_scans))
    val_dataset = val_dataset.shuffle(len(val_scans))
    test_dataset = test_dataset.shuffle(len(test_scans))

    train_dataset = train_dataset.map(process_path)
    val_dataset = val_dataset.map(process_path)
    test_dataset = test_dataset.map(process_path)

    for scan, label in train_dataset.take(1):
        display([tf.squeeze(scan), tf.argmax(label, axis=-1)])
        print(scan.shape)
        print(label.shape)

    model = improved_unet(4)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    loss_accuracy = model.fit(train_dataset.batch(10), epochs=1, validation_data=val_dataset.batch(10))

    labels = show_predictions(test_dataset, model)
    print("Layer 0: ", dsc(labels[0], labels[1], 0))
    print("Layer 1: ", dsc(labels[0], labels[1], 1))
    print("Layer 2: ", dsc(labels[0], labels[1], 2))
    print("Layer 3: ", dsc(labels[0], labels[1], 3))

##    test_brain = Image.open(str(test_scans[0]))
##    test_brain = np.asarray(test_brain, dtype=np.uint8)
##    print(test_brain.shape)

if __name__ == "__main__":
    main()
