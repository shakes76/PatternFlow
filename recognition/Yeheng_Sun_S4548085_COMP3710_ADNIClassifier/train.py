# import package
from operator import mod
from tensorflow import keras
import tensorflow_addons as tfa
import pandas as pd
import warnings
import module
import dataset
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import matplotlib.pyplot as plt

# ignore warning
warnings.filterwarnings('ignore')

class_n = 1 # number of classes 0,1
input_shape = (256, 256, 3) # input shape of the model
image_size = 256 
lr = 0.0003  # learning rate
weight_decay = 0.0001 # weight decay
batch_size = 64 # batch size
epoch_n = 135  # number of epochs
patch_len = 45  # Size of the patches to be extract from the input images
patch_n = (image_size // patch_len) ** 2
proj_vec_n = 16
head_n = 4
transformer_n = [
    proj_vec_n * 2,
    proj_vec_n,
]  # Size of the transformer layers
transformer_layer_n = 4
layer_list = [1024,512]  # Number of neurons in MLP

# train and validation tensorflow dataset 
train_ds = dataset.createTrainData(dataset.img_size, batch_size)



def train(model):
    # define adam optimizer
    optimizer = tfa.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay
    )

    # define loss and metric of the model
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(from_logits=True),
        metrics=[
            BinaryAccuracy(name="accuracy"),
        ],
    )

    # define checkpoint
    checkpoint_filepath = "./tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                  patience=3, min_delta=0.001)

    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epoch_n,
        callbacks=[checkpoint_callback, reduce_lr],
    )

    model.load_weights(checkpoint_filepath)
    model.save_weights("model.h5")
    return history

def plot_loss(model):
    history = pd.DataFrame(model.history)
    plt.plot(history.index,history['loss'],label='Train Loss')
    plt.plot(history.index,history['val_loss'],label='Validation Loss')
    plt.legend()
    plt.show()

def plot_acc(model):
    history = pd.DataFrame(model.history)
    plt.plot(history.index,history['accuracy'],label='Train Accuracy')
    plt.plot(history.index,history['val_accuracy'],label='Validation Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # define vision transformer model
    vision_transformer = module.vision_transformer(input_shape, patch_len, patch_n, proj_vec_n, transformer_layer_n,
                                                   head_n, class_n, transformer_n, layer_list)
    # train model
    model_history = train(vision_transformer)

    # visualization
    plot_acc(model_history)
    plot_loss(model_history)
    
