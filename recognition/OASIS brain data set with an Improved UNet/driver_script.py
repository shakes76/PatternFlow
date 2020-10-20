# This is the test driver script.
# This script shows example usage of the module created in solution.py
# It creates relevant plots and visualisations


# include a main method
# can run the solution

# can use numpy if needed.

# want to show an example of my solution being run.

# import the correct modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import zipfile
import glob
from IPython.display import clear_output

from solution import unet_model


def decode_png(file_path):
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png, channels=1)
    png = tf.image.resize(png, (256, 256))
    return png


def process_path(image_fp, mask_fp):
    image = decode_png(image_fp)
    image = tf.cast(image, tf.float32) / 255.0
    
    mask = decode_png(mask_fp)
    mask = mask == [0, 85, 170, 255]
    return image, mask


def import_ISIC_data():
    """ Download the dataset """

    # Download data
    dataset_url = "https://cloudstor.aarnet.edu.au/sender/download.php?token=505165ed-736e-4fc5-8183-755722949d34&files_ids=10012238"
    data_path = tf.keras.utils.get_file(origin=dataset_url, fname="ISIC2018_Task1-2_Training_Data.zip")
    
    # extract the zip file
    with zipfile.ZipFile(data_path) as masks_zf:
        masks_zf.extractall()
    
    # List files
    images = sorted(glob.glob("ISIC2018_Task1_Training_Input_x2/*.png"))
    masks = sorted(glob.glob("ISIC2018_Task1_Training_GroundTruth_x2/*.png"))

    # split image and masks into training, validating and testing sets
    train_images, val_images, test_images = np.split(images, 3)    
    train_masks, val_masks, test_masks = np.split(masks, 3)
      
    # make dataset from images and masks
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

    # shuffle dataset
    train_ds = train_ds.shuffle((len(train_images)))
    val_ds = val_ds.shuffle((len(val_images)))
    test_ds = test_ds.shuffle((len(test_images)))
    
    # map the dataset to process_path function
    train_ds = train_ds.map(process_path)
    val_ds = val_ds.map(process_path)
    test_ds = test_ds.map(process_path)
        
    return train_ds, val_ds, test_ds
    

def import_OASIS_data():
    """ Download the dataset """

    # Download the dataset
    dataset_url = "https://cloudsor.aarnet.edu.au/............."
    data_path = tf.keras.utils.get_file(origin=dataset_url, fname="content/keras_png_slices_data.zip")
    
    with zipfile.ZipFile(data_path) as zf:
        zf.extractall()
        
    # List files
    train_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_train/*.png"))
    train_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_train/*.png"))
    val_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_validate/*.png"))
    val_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_validate/*.png"))
    test_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_test/*.png"))
    test_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_test/*.png"))

    return train_images, train_masks, val_images, val_masks, test_images, test_masks  
    
    
def display(display_list):
    plt.figure(figsize=(10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(tf.reshape(images[i], (h,w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def show_predictions(model, ds, num=1):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.argmax(pred_mask[0], axis=1)
        display([tf.squeeze(image), tf.argmax(mask, axis=1), pred_mask])


class DisplayCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, val_ds, logs=None):
        clear_output(wait=True)
        show_predictions(val_ds)


def analyse_training_history(history):
    
    # analyse the model
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    
    return 0


def analyse_predictions(predictions, X_test, Y_test, target_names):
    
    # determine if predictions were correct
    correct = (predictions == Y_test)

    # number of images testes
    total_test = len(X_test)

    print("Total Tests:", total_test)
    print("Predictions:", predictions)
    print("Which Correct:", correct)
    print("Total Correct:", np.sum(correct))
    print("Accuracy:", np.sum(correct)/total_test)

    print(classification_report(Y_test, predictions, target_names=target_names))
    
    return 0


def main():
    
    # import the data
    train_ds, val_ds, test_ds = import_ISIC_data()
    return 0 ##################################################  
        
    # plot example image
    for image, mask in train_ds.take(1):
        display([tf.squeeze(image), tf.argmax(mask, axis=1)])
    
    # create the model
    model = unet_model(4, f=4)
    
    # compile the model
    model.compile(optimizer='adam', loss='catagorical_crossentropy', metrics=['accuracy'])
    
    # show predictions
    show_predictions(val_ds)
    
    #
    history = model.fit(train_ds.batch(32), epochs=3, validation_data=val_ds.batch(32), callbacks=[DisplayCallback()])
    
    #
    show_predictions(test_ds, 3)

    
    
    # analyse history of training the model
    #analyse_training_history(history)
    
    # make predictions using the model
    #predictions = model.predict(X_test)
    #predictions = np.argmax(predictions, axis=1)
    
    # show plots of some of these predictions
    
    
    # numerically analyse the performance model
    #analyse_predictions(predictions, X_test, Y_test, target_names)
    
    return 0
    
    

if __name__ == "__main__":
    main()