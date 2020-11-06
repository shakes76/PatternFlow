#!/usr/bin/env/python

"""
Driver script for ISICs UNet recognition problem.

Created by Christopher Bailey (45576430) for COMP3710 Report.
"""


import tensorflow as tf
from isicsunet import IsicsUnet


def main():
    print(tf.__version__)

    # TensorFlow provided code to limit GPU memory growth
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

    model = IsicsUnet()

    model.load_data()
    #model.visualise_loaded_data()

    model.build_model()
    model.model.summary()

    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                        loss='binary_crossentropy',
                        metrics=['accuracy', IsicsUnet.dice_coefficient])
    #model.show_predictions()

    history = model.model.fit(x=model.train_ds.batch(1),
                              validation_data=model.val_ds.batch(1),
                              epochs=10)
    model.show_predictions()

    print("Evaluate")
    result = model.evaluate(model.test_ds)
    print(result)

    print("END")


if __name__ == "__main__":
    main()
