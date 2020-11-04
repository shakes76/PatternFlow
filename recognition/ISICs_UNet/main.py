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
    print(model.train_ds)
    print(model.val_ds)
    #model.visualise_loaded_data()

    model.build_model()
    model.model.summary()

    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.9),
                        #loss='binary_crossentropy',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    model.show_predictions()

    history = model.model.fit(x=model.train_ds.batch(model.BATCH_SIZE, drop_remainder=True),
                              validation_data=model.val_ds.batch(model.BATCH_SIZE, drop_remainder=True),
                              verbose=1,
                              epochs=1)
    model.show_predictions()

    # todo test and compare to dice similarity coefficient

    print("END")


if __name__ == "__main__":
    main()
