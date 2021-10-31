import tensorflow as tf
from driver import pre_process
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from model import *

IMG_WIDTH = 256
IMG_HEIGHT = 192
IMG_CHANNELS = 3

def main():
    print(tf.__version__)
    # use batch size of 2 to save VRAM
    BATCH_SIZE = 2

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
    pre_p = pre_process()
    pre_p.load_data()
    # model.visualise_loaded_data()
    input = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    pre_p.Improved_unet(input)
    pre_p.model.summary()

    pre_p.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss=pre_process.dice_loss,
                        metrics=pre_process.dice_coefficient)
    # pre_p.show_predictions()  # sanity check

    history = pre_p.model.fit(x=pre_p.train_ds.batch(BATCH_SIZE),
                              validation_data=pre_p.val_ds.batch(BATCH_SIZE),
                              epochs=15)
    pre_p.show_predictions()
    
    result = pre_p.model.evaluate(pre_p.test_ds.batch(BATCH_SIZE))
    print(dict(zip(pre_p.model.metrics_names,result)))
    print("END")
    
if __name__ == "__main__":
    main()