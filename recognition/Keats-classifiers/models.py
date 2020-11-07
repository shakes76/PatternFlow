import tensorflow as tf

def generate_unet(shape, data):
    pass

def generate_knee_model(data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(32,)), # wrong,
        tf.keras.layers.Conv2D(32, filter_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),

        tf.keras.layers.Conv2D(64, filter_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'), # 10 class classifier
    ])

    return model

