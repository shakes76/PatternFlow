#!/usr/bin/env/python

"""
Driver script for ISICs UNet recognition problem.

Created by Christopher Bailey (45576430) for COMP3710 Report.

Segments of code in this file are based on code from TensorFlow documentation pages
"""


import tensorflow as tf
from isicsunet import IsicsUnet


def main():
    print(tf.__version__)

    # use batch size of 1 to save VRAM
    BATCH_SIZE = 1

    # TensorFlow provided code to limit GPU memory growth
    # Retrieved from:
    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Initialise and load data into model
    model = IsicsUnet()
    model.load_data()
    #model.visualise_loaded_data()  # sanity check

    # Set up model
    model.build_model()
    model.model.summary()
    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                        loss=IsicsUnet.dice_loss,
                        metrics=IsicsUnet.dice_coefficient)
    #model.show_predictions()  # sanity check

    # Train model
    history = model.model.fit(x=model.train_ds.batch(BATCH_SIZE),
                              validation_data=model.val_ds.batch(BATCH_SIZE),
                              epochs=3)
    model.show_predictions()

    # Get dice similarity for test set and show result
    print("Evaluate")
    result = model.model.evaluate(model.test_ds.batch(BATCH_SIZE))
    print(dict(zip(model.model.metrics_names,result)))

    print("END")


if __name__ == "__main__":
    main()
