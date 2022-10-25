import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import module
import dataset
warnings.filterwarnings('ignore')


num_classes = 2

input_shape = (dataset.image_size, dataset.image_size, 3)
image_size = module.image_size
learning_rate = 0.00003
weight_decay = 0.0001
batch_size = 64
num_epochs = 1  # We'll resize input images to this size 136
patch_size = 20  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 126
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [4096,2048]  # Size of the dense layers of the final classifier


#x_train, y_train, x_test, y_test = dataset.prepareData()
train_ds = dataset.createTrainData(image_size,batch_size)
validate_ds = dataset.createTestData(image_size,batch_size)
module.dataAugmentation.layers[0].adapt(train_ds)
# print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    checkpoint_filepath = "./tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
        validation_data=validate_ds,
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(validate_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    model.save_weights("model.h5")
    return history

# def plot_loss(model_history):
#     for k,v in model_history.items():
#         loss = []
#         val_loss = []
#         loss.append(v.model_history['loss'][:])
#         val_loss.append(v.model_history['val_loss'][:])
#         plt.figure(figsize = (15, 6))
#         plt.plot(np.mean(loss, axis=0))
#         plt.plot(np.mean(val_loss, axis=0))
#         plt.yscale('log')
#         plt.yticks(ticks=[1,1e-1,1e-2])
#         plt.xlabel('Epochs')
#         plt.ylabel('Average Logloss')
#         plt.legend(['Training','Validation'])
#         plt.show()

transformer_model = module.createModel(input_shape,patch_size,num_patches,projection_dim,transformer_layers,num_heads,num_classes,transformer_units,mlp_head_units)
model_history = run_experiment(transformer_model)



hist_df = pd.DataFrame(model_history.history) 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
# plot_loss()
