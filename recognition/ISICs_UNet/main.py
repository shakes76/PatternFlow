#!/usr/bin/env/python

"""
Driver script for ISICs UNet recognition problem.

Created by Christopher Bailey (45576430) for COMP3710 Report.
"""


import tensorflow as tf
from isicsunet import IsicsUnet

def main():
    print(tf.__version__)

    model = IsicsUnet()

    model.load_data()

    model.build_model()

    model.show_predictions()

    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

    history = model.fit(x=model.train_ds,
                        validation_data=model.val_ds,
                        epochs=3)

    model.show_predictions()

    print("END")


if __name__ == "__main__":
    main()
