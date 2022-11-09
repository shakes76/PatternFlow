import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from model import improved_unet
from sklearn.utils import shuffle

"""
This function decodes a PNG file into a tensor. 
Parameters: 
    file_path: File path of the PNG image
Returns: 
    Tensor representation of the image
"""
def decode_png(file_path):
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png, channels=1)
    png = tf.image.resize(png, (256, 256))
    return png

"""
This function decodes a JPEG file into a tensor. 
Parameters: 
    file_path: File path of the JPEG image
Returns: 
    Tensor representation of the image
"""
def decode_jpeg(file_path):
    jpeg = tf.io.read_file(file_path)
    jpeg = tf.image.decode_jpeg(jpeg, channels=1)
    jpeg = tf.image.resize(jpeg, (256, 256))
    return jpeg

"""
Processes the file path for both of the scan and label images to 
convert to tensors
Paremeters: 
    scan_fp: string representation for the file path of the scan image
    label_fp: string represntation for the file path of the label image
Returns: 
    A tuple containing the tensors for the scan image and the label images
         (scan tensor, label tensor)
"""
def process_path(scan_fp, label_fp):
    #Process the Skin scan
    scan = decode_jpeg(scan_fp)
    scan = tf.cast(scan, tf.float32) / 255.0

    #Process the Label image
    label = decode_png(label_fp)
    label = label == [0, 255]
    return scan, label

"""
Gives an image visualisation for each tensor contained in the display_list.
Parameters: 
    display_list: tensors to display
Returns:
"""
def display(display_list):
    plt.figure(figsize=(10,10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

"""
Displays the predictions of the given model and dataset. The output 
is visualised as original photo, ground truth segmenation and predicted
segmentation.
Parameters:
    dataset: the dataset to take the images from
    model: the model used to predict the segmentation for images from the dataset
Returns:
    A tuple containing the ground truth and predicted label.
"""
def show_predictions(dataset, model):
    num = 1
    for scan, label in dataset.take(num):
        pred_label = model.predict(scan[tf.newaxis, ...])
        pred_label = tf.argmax(pred_label[0], axis=-1)
        display([tf.squeeze(scan), tf.argmax(label, axis=-1), pred_label])
    return (label, pred_label)

"""
Computes the dice coefficient between the ground truth label image
and the predicted label given the layer. This is an adaption of a 
solution by Zabir Al. Nazi (2020)(see references in README.md).
Parameters:
    label: ground truth label image
    pred_label: predicted label image
    layer: layer to compute the dice coefficient for. The
           value must be either 0 or 1.
Returns:
    The dice coefficient between the two labels, i.e. how much the
    two images overlap given the layer.
"""
def dice_function(label, pred_label, layer):
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

"""
Modulised function to display the Loss and Accuracy graphs for the
training of the model.
Parameters: 
    metrics: The loss and accuracy metrics calculated during the training
             of the model.
"""
def show_loss_and_accuracy(metrics):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(metrics.history['loss'], label='Training Loss')
    plt.plot(metrics.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.subplot(1, 2, 2)
    plt.plot(metrics.history['accuracy'], label='Training Accuracy')
    plt.plot(metrics.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Graph')
    plt.show()


"""
Main function to run the scipt.
"""
def main():
    number_Of_output_channels = 2
    number_of_training_epochs = 50
    batch_size = 10
    #List the file paths of the data
    skin_scans = sorted(glob.glob('ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/*.jpg'))
    labels = sorted(glob.glob('ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/*.png'))

    print(len(skin_scans))
    print(len(labels))

    #Shuffle the data before the split
    skin_scans, labels = shuffle(skin_scans, labels)

    #Split the data into train, validate and test sets
    #Note that we did a 85% train, 15% val and 5% test split
    #See README.md for more information.
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

    train_dataset = train_dataset.map(process_path)
    val_dataset = val_dataset.map(process_path)
    test_dataset = test_dataset.map(process_path)

    #Confirm that the data has been loaded correctly.
    for scan, label in train_dataset.take(1):
        display([tf.squeeze(scan), tf.argmax(label, axis=-1)])
        print(scan.shape)
        print(label.shape)

    
    model = improved_unet(number_Of_output_channels, f=4)

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    loss_accuracy = model.fit(train_dataset.batch(batch_size), epochs=number_of_training_epochs, validation_data=val_dataset.batch(batch_size))

    show_loss_and_accuracy(loss_accuracy)
    
    #Compute the dice coefficients for each test image.
    num_of_tests = 130
    i = 1
    layer_0_average = 0
    layer_1_average = 0
    for scan, label in test_dataset.take(num_of_tests):
        pred_label = model.predict(scan[tf.newaxis, ...])
        pred_label = tf.argmax(pred_label[0], axis=-1)
        print("Displaying Dice Coefficient for test: ", i)
        layer_0_dsc = dice_function(label, pred_label, 0)
        layer_1_dsc = dice_function(label, pred_label, 1)
        print("Layer 0: ", layer_0_dsc)
        print("Layer 1: ", layer_1_dsc)
        i += 1
        layer_0_average += layer_0_dsc
        layer_1_average += layer_1_dsc
        display([tf.squeeze(scan), tf.argmax(label, axis=-1), pred_label])
    layer_0_average = layer_0_average / 130
    layer_1_average = layer_1_average / 130
    print("Average Layer 0 DSC: ", layer_0_average)
    print("Average Later 1 DSC: ", layer_1_average)

#Run the main function.
if __name__ == "__main__":
    main()
