from modules import *
from dataset import *

def train_unet(train_x, train_y, valid_x, valid_y):
    """
    Trains the model on the training data and returns results

    :return: results of training
    """
    # %%
    unet = unet_full(input_size=(128,128,3), n_filters=32, n_classes=255)
    unet.summary()
    unet.compile(optimizer=tf.keras.optimizers.Adam(), 
             #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

    # %%
    results = unet.fit(train_x, train_y, batch_size=32, epochs=20, validation_data=(valid_x, valid_y))
    return unet, results