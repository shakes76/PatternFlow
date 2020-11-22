import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import glob
import sys
from cnn import ConvBlock, CNNModel

def main(arglist):
    #Parameters
    n_epochs = 10
    batch_size = 32
    learning_rate = 1E-3

    #Specify directory of data from arglist input
    data_dir = arglist[0]
        
    #Load all the filenames
    filenames = glob.glob(os.path.join(data_dir, '*.png'))

    #Split filenames by patiend id
    patient_files = dict()
    for fn in filenames:
        pid = fn.split(os.path.sep)[-1].split('_')[0]
        if pid not in patient_files:
            patient_files[pid] = [fn]
        else:
            patient_files[pid].append(fn)

    #Count number of patients
    patient_ids = list(patient_files.keys())
    patient_count = len(patient_ids)


    #Split datasets (validation, test, training)
    random.seed(123)
    not_valid = True
    while not_valid:
        random.shuffle(patient_ids)

        #Split the patients - 16% validation, 20% test, 64% training
        val_count = int(patient_count * 0.16)
        test_count = int(patient_count * 0.36)
        val_patients = patient_ids[:val_count]
        test_patients = patient_ids[val_count:test_count]
        train_patients = patient_ids[test_count:]

        val_images = []
        test_images = []
        train_images = []

        #Add every patient's filenames to their respective image datasets
        for pid in val_patients:
            val_images.extend(patient_files[pid])
        for pid in test_patients:
            test_images.extend(patient_files[pid])
        for pid in train_patients:
            train_images.extend(patient_files[pid])

        #Extract labels
        train_labels = [fn.replace('R_I_G_H_T', 'right').replace('L_E_F_T', 'left').split(os.path.sep)[-1].split('_')[-2].split('.')[0].lower()
                for fn in train_images]
        val_labels = [fn.replace('R_I_G_H_T', 'right').replace('L_E_F_T', 'left').split(os.path.sep)[-1].split('_')[-2].split('.')[0].lower() 
                for fn in val_images]
        test_labels = [fn.replace('R_I_G_H_T', 'right').replace('L_E_F_T', 'left').split(os.path.sep)[-1].split('_')[-2].split('.')[0].lower() 
                for fn in test_images]
        not_valid = False #Break while loop

        #Count number of images in each dataset labelled 'right'
        train_right_count = 0
        for label in train_labels:
            if label == 'right':
                train_right_count += 1

        val_right_count = 0
        for label in val_labels:
            if label == 'right':
                val_right_count += 1

        test_right_count = 0
        for label in test_labels:
            if label == 'right':
                test_right_count += 1

        #Ratios of 'right' labelled to 'left' labelled
        right_ratios = [train_right_count/len(train_labels), val_right_count/len(val_labels), test_right_count/len(test_labels)]

        #If too many or too few 'right' images, re-shuffle and re-split datasets.
        for ratio in right_ratios:
            if ratio > 0.7 or ratio < 0.3:
                print("Re-shuffling")
                not_valid = True #continue while loop
    
    #Class names list and number of classes
    class_names = sorted(set(train_labels))
    # print(class_names)
    num_classes = len(class_names)

    #Create tensorflow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    #Map filenames and labels to data arrays
    img_height = 228
    img_width = 260
    def map_fn(filename, label):
        # Load the raw data from the file as a string.
        img = tf.io.read_file(filename)
        # Convert the compressed string to a 3D uint8 tensor.
        img = tf.image.decode_jpeg(img, channels=1) # channels=3 for RGB, channels=1 for grayscale
        # Resize the image to the desired size.
        img = tf.image.resize(img, (img_height, img_width))
        # Standardise values to be in the [0, 1] range.
        img = tf.cast(img, tf.float32) / 255.0
        # One-hot encode the label.
        one_hot = tf.cast(label == class_names, tf.uint8)
        # Return the processed image and label.
        return img, one_hot

    #Apply mapping to datasets
    train_ds = train_ds.map(map_fn)
    val_ds = val_ds.map(map_fn)
    test_ds = test_ds.map(map_fn)

    #Configure dataset for performance and shuffling. Shuffle buffer = number of images
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(len(train_images))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.shuffle(len(val_images))
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache()
    test_ds = test_ds.shuffle(len(test_images))
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    #Create ModelCheckpoint to save and load trained models
    checkpoint_path = "training/ckpt01.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq='epoch'
    )

    #Create an instance of the model
    model = CNNModel(num_classes=num_classes)

    #Use the model to predict an image to automatically 'build' it and allow calling of model.summary()
    image_batch, label_batch = next(iter(val_ds.take(1)))
    model.predict(image_batch)

    model.summary()

    #Loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), #Hyperparameter
        loss=loss_fn,
        metrics=['accuracy']
    )

    #Load saved weights
    # model.load_weights(checkpoint_path).expect_partial()

    #Train the model
    results = model.fit(train_ds, epochs=n_epochs, callbacks=[cp_callback], validation_data=val_ds)

    #Check final weights by evaluating on all sets
    print("Training set: ")
    model.evaluate(train_ds, verbose=2)
    print("Validation set: ")
    model.evaluate(val_ds, verbose=2)
    print("Test set: ")
    model.evaluate(test_ds, verbose=2)

    Plot accuracy vs validation accuracy and loss vs validation loss (for every epoch)
    plt.plot(results.history['accuracy'], label='accuracy')
    plt.plot(results.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    # plt.ylim(0.5, 1.0)
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

    plt.plot(results.history['loss'], label="loss")
    plt.plot(results.history['val_loss'], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    # plt.ylim(0.0, 1.0)
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
