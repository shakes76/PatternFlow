import tensorflow as tf
print(tf.__version__)


# Function for model
def classification(images_shape):
    """Model for binary classification 
    Parameters: 
          images_shape (tuple): The input shape of the images used to train the model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape=images_shape),
        #tf.keras.layers.Conv2D(128,(3,3), activation='relu',padding='same'),
        #tf.keras.layers.MaxPooling2D((2,2)),
        #tf.keras.layers.Dropout(0.25),

        #tf.keras.layers.Conv2D(32,(3,3), activation='relu',padding='same'),
        #tf.keras.layers.Conv2D(256,(3,3), activation='relu',padding='same'),
        #tf.keras.layers.MaxPooling2D((2,2)),
        #tf.keras.layers.Dropout(0.25),

        #tf.keras.layers.Conv2D(64,(3,3), activation='relu',padding='same'),
        #tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
#activation sigmoid and corresponding loss function