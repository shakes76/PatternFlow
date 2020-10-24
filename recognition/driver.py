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

def decode_jpeg(file_path):
    jpeg = tf.io.read_file(file_path)
    jpeg = tf.image.decode_jpeg(jpeg, channels=1)
    jpeg = tf.image.resize(jpeg, (256, 256))
    return jpeg

def process_path(scan_fp, label_fp):
    #Process the Skin scan
    scan = decode_jpeg(scan_fp)
    scan = tf.cast(scan, tf.float32) / 255.0

    #Process the Label image
    label = decode_png(label_fp)
    label = label == [0, 255]
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
    for scan, label in dataset.take(num):
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
    print(pred_label.shape)
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

def newDSC(label, pred_label, layer):
    label = np.argmax(label, axis=-1)
    label = tf.reshape(label, [-1])
    pred_label = tf.reshape(pred_label, [-1])
    label = label.numpy()
    pred_label = pred_label.numpy()
    label = label == layer
    pred_label = pred_label == layer
    intersection = np.sum(label * pred_label)
    label_card = np.sum(label)
    pred_label_card = np.sum(pred_label)
    numerator = 2 * intersection
    demonimator = label_card + pred_label_card
    return numerator / demonimator




def main():
    #Get the file location of where the data is stored.
    data_dir = pathlib.Path('H:/Year 3/Sem 2/COMP3710/Report/PatternFlow/recognition/ISIC2018_Task1-2_Training_Data')
    #The following 5 lines were used to download and unzip the data, these have been commented out for faster debugging.
    #However, they have been left in to show how the files were retreived.
    #data_dir_tf = tf.keras.utils.get_file(origin='https://cloudstor.aarnet.edu.au/sender/?s=download&token=505165ed-736e-4fc5-8183-755722949d34',
    #                                      fname='H:/Year 3/Sem 2/COMP3710/Report/PatternFlow/recognition/ISIC2018_Task1-2_Training_Data.zip')

    #with zipfile.ZipFile(data_dir_tf) as zf:
    #   zf.extractall()
    number_Of_output_channels = 2
    #List the file paths of the data
    skin_scans = sorted(glob.glob('ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/*.jpg'))
    labels = sorted(glob.glob('ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/*.png'))

    print(len(skin_scans))
    print(len(labels))

    #Split the data into train, validate and test sets
    train_ds_size = 2204
    val_ds_size = 260
    train_scans = skin_scans[:train_ds_size]
    train_labels = labels[:train_ds_size]
    val_scans = skin_scans[train_ds_size:train_ds_size + val_ds_size]
    val_labels = labels[train_ds_size:train_ds_size + val_ds_size]
    test_scans = skin_scans[train_ds_size + val_ds_size:]
    test_labels = labels[train_ds_size + val_ds_size:]

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

    
    model = improved_unet(number_Of_output_channels, f=8)

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    loss_accuracy = model.fit(train_dataset.batch(10), epochs=2, validation_data=val_dataset.batch(10))

    labels = show_predictions(test_dataset, model)
    print("------------------------------")
    print("New DSC Calculations:")
    print("Layer 0: ", newDSC(labels[0], labels[1], 0))
    print("Layer 1: ", newDSC(labels[0], labels[1], 1))
    """
    To add: 
    - need to show more predictions
    - Improve the efficiency of DSC to caclulcate the DSC on all test images.
    - Include the plots for loss and accuracy.
    """

##    test_brain = Image.open(str(test_scans[0]))
##    test_brain = np.asarray(test_brain, dtype=np.uint8)
##    print(test_brain.shape)


if __name__ == "__main__":
    main()
