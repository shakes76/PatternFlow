import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from modules import create_tuned_classifier
from dataset import *

tuner = kt.Hyperband(create_tuned_classifier,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='tuning',
                     project_name='Transformer')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(get_train_data(), epochs=100, validation_data=get_valid_data(), callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.\n
Learning Rate:{best_hps.get('learning_rate')}\n
Transformers:{best_hps.get('transformers')}\n
Dropout:{best_hps.get('dropout')}\n
""")

