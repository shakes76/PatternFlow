import tensorflow as tf
def create_model():
    covn_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    covn_base.trainable = True
    # Freeze the front layer and train the last seven layers
    for layers in covn_base.layers[:-7]:
        layers.trainable = False
    # Building models
    model = tf.keras.Sequential()
    model.add(covn_base)
    model.add(tf.keras.layers.GlobalAveragePooling2D())  # Adding global average pooling layer
    model.add(tf.keras.layers.Dense(512, activation='relu'))  # Add full connectivity layer
    model.add(tf.keras.layers.Dropout(rate=0.5))  # Add dropout layer to prevent over fitting
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Add output layer(2 categories)
    model.summary()  # Print parameter information of each layer

    # Compiling model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  # Using Adam optimizer, the learning rate is 0.001
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # Cross entropy loss function
                  metrics=["accuracy"])  # evaluation function
    return model
