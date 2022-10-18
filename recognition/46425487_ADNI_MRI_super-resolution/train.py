import tensorflow as tf
from tensorflow import keras
from modules import *
from dataset import *

train_ds, valid_ds, test_ds = process_data()

early_stopper= keras.callbacks.EarlyStopping(monitor="loss", patience=10)

checkpoint_filepath = "./weights"

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

model = get_model()
model.summary()

callbacks = [early_stopper, model_checkpoint]
loss_function= keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss_function)

result = model.fit(train_ds, epochs=100, callbacks=callbacks, validation_data=valid_ds, verbose=2)

plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.savefig("acc.jpg")

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("loss.jpg")