import tensorflow as tf

from dataset import get_data_preprocessing
from modules import get_model
from utils import *

if __name__ == "__main__":
    cropped_image_size = (192, 160)
    epochs = 5

    train_ds, test_ds, preprocessing = get_data_preprocessing(
        image_size=(240, 256), cropped_image_size=cropped_image_size, cropped_pos=(20, 36))
    model = get_model(max(cropped_image_size))

    model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

    model.fit(x=train_ds,
          epochs=epochs,
          validation_data=test_ds)

    model.save_weights(DATA_DIR + "checkpoints/my_checkpoint")