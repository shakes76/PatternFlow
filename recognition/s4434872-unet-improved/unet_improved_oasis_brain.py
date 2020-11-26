"""
OASIS Brain Dataset Segmentation with Improved UNet, 
with all labels having a minimum Dice Similarity Coefficient 
of 0.9 on the test set.

Driver script.

@author Dhilan Singh (44348724)

Start Date: 01/11/2020
"""
import tensorflow as tf
from IPython.display import clear_output
import glob
import preprocess
import model as mdl
import metrics
import visualisation


def main():
    """
    Main function.
    """
    print('Tensorflow Version:', tf.version.VERSION)

    # Download the dataset (use the direct link given on the page)
    print("> Loading images ...")
    # tf.keras.utils.get file downloads a file from a URL if it not already in the cache.
    #     origin: Original URL of the file.
    #     fname: Name of the file. If an absolute path /path/to/file.txt is specified the 
    #            file will be saved at that location (in cache directory).
    #            NEEDS FILE EXTENSION TO WORK!!!
    #     extract: If true, extracting the file as an Archive, like tar or zip.
    #     archive_format: zip, tar, etc...
    #     returns: Path to the downloaded file.
    dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
    data_path = tf.keras.utils.get_file(origin=dataset_url,
                                        fname="keras_png_slices_data.zip",
                                        extract=True,
                                        archive_format="zip")

    # Remove the .zip file extension from the data path
    data_path_clean = data_path.split('.zip')[0]

    # Load filenames into a list in sorted order
    train_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_train/*.png"))
    train_masks = sorted(glob.glob(data_path_clean +"/keras_png_slices_seg_train/*.png"))
    val_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_validate/*.png"))
    val_masks = sorted(glob.glob(data_path_clean + "/keras_png_slices_seg_validate/*.png"))
    test_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_test/*.png"))
    test_masks = sorted(glob.glob(data_path_clean + "/keras_png_slices_seg_test/*.png"))

    # Build tf datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

    # Make the dataset to be reshuffled each time it is iterated over.
    # This is so that we get different batches for each epoch.
    # For perfect shuffling, the buffer size needs to be greater than or equal to 
    # the size of the dataset.
    train_ds = train_ds.shuffle(len(train_images))
    val_ds = val_ds.shuffle(len(val_images))
    test_ds = test_ds.shuffle(len(test_images))

    # Use Dataset.map to apply preprocessing transformation.
    # Normalize the images and pixel-wise one-hot encode the segmentation masks.
    print("> Preprocessing images ...")
    train_ds = train_ds.map(preprocess.process_path)
    val_ds = val_ds.map(preprocess.process_path)
    test_ds = test_ds.map(preprocess.process_path)

    # Input Image Parameters for model
    image_pixel_rows = 256 
    image_pixel_cols = 256
    image_channels = 1
    
    # Create Improved UNet Model
    print("> Building Model ...")
    model = mdl.improved_unet_model(4, n_filters=16, input_size=(image_pixel_rows, image_pixel_cols, image_channels))

    # Compile Model using DSC as loss function and a metric
    model.compile(optimizer='adam',
                loss=metrics.dice_coefficient_loss,
                metrics=['accuracy', metrics.dice_coefficient])
    model.summary()

    # Training Hyperparameters
    BATCH_SIZE = 12
    EPOCHS = 8

    # Fill in some of the blank by default callback functions
    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            visualisation.show_predictions(model, val_ds)
        # can fill in another function on_epoch_start() if need
        # to perform some action at the start of an epoch.

    # Train model for epochs=EPOCHS with data batched as BATCH_SIZE
    print("> Start Training Model ...")
    history = model.fit(train_ds.batch(BATCH_SIZE), epochs=EPOCHS,
                        validation_data=val_ds.batch(BATCH_SIZE),
                        callbacks=[DisplayCallback()])
    print("> Training Finished")

    # Plot training and validation results
    visualisation.visualise_training(history, EPOCHS)

    # Evaluate trained model on the test set
    print("> Evaluating Trained Model on Test Set ...")
    test_DSC_loss, test_acc, test_DSC = model.evaluate(test_ds.batch(BATCH_SIZE), verbose=2)

    # Display Test Set Results (Average over batches)
    print("----- Test Set Results -----")
    print("DSC Loss: ", test_DSC_loss)
    print("DSC: ", test_DSC)
    print("Accuracy: ", test_acc)

    # Show some test set predictions
    print("> Showing Some Test Set Predictions ...")
    visualisation.save_predictions(model, test_ds, 3)

    # End of operation
    print('End')



if __name__ == "__main__":
    # Enter main loop (not really a loop though)
    main()
