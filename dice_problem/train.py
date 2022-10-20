from modules import *
from dataset import *

def train_unet(train_x, train_y, valid_x, valid_y):
    # %%
    unet = unet(input_size=(128,128,3), n_filters=32, n_classes=255)
    unet.summary()
    unet.compile(optimizer=tf.keras.optimizers.Adam(), 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

    # %%
    results = unet.fit(train_x, train_y, batch_size=32, epochs=20, validation_data=(valid_x, valid_y))
    return results