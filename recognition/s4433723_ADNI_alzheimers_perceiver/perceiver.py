"""
REFERENCE FOR PERCEIVER HELP ON CIFAR10
https://keras.io/examples/vision/perceiver_image_classification/
"""
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# check tf version etc.
print("Tensorflow version: ", tf.__version__)

# =============== PREPARE DATA ===============

DATASET_DIR = 'datasets/'
OUTPUT_DIR = 'output/'

# should get this from directory
INPUT_DS_SIZE = 18680
# size to resize input images
IMG_SIZE = 8

# load AKOA dataset from processed datasets directory
akoa_ds_tuple = tf.keras.preprocessing.image_dataset_from_directory(directory=DATASET_DIR,
                                                                    shuffle=True,
                                                                    seed=999,
                                                                    image_size=(IMG_SIZE, IMG_SIZE),
                                                                    batch_size=INPUT_DS_SIZE,
                                                                    labels="inferred",
                                                                    label_mode="categorical",
                                                                    color_mode="grayscale",
                                                                    ),


# extract dataset from tuple
akoa_ds = akoa_ds_tuple[0]
assert isinstance(akoa_ds, tf.data.Dataset)

# normalise dataset
data_augmenter = tf.keras.Sequential([
    #tf.keras.layers.experimental.preprocessing.Normalization(),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5),
    tf.keras.layers.experimental.preprocessing.RandomContrast(1),
])
akoa_ds = akoa_ds.map(lambda x, y: (data_augmenter(x), y))

# get data into numpy arrays
x_data, y_data = next(iter(akoa_ds))
x_data = np.array(x_data)
y_data = np.array(y_data)

# display e.g. image from ds
first_image = x_data[0, :, :, 0]
print(first_image.shape)
plt.imsave(OUTPUT_DIR + "eg_img.png", first_image, format="png", cmap=plt.cm.gray)

# train test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

print("Input Shapes:")
print("X train: ", x_train.shape)
print("Y train: ", y_train.shape)
print("X test: ", x_test.shape)
print("Y test: ", y_test.shape)

# =============== CONFIGURE HYPERPARAMETERS ===============

NUM_CLASSES = 2
NUM_CHANNELS = 1
LEARN_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 1
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
PATCH_SIZE = 2  # Size of patches to be extracted from input images.
PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # Size of the img data array.
LATENT_ARRAY_SIZE = 128  # Size of the latent array.
PROJECTION_SIZE = 128  # Embedding size of each element in the data and latent arrays.
NUM_HEADS = 8  # Number of transformer heads.

# Size of the Transformer Dense network.
dense_units = [PROJECTION_SIZE, PROJECTION_SIZE]
num_transformer_blocks = 4
num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.

# Size of the Feedforward network of the final classifier.
classifier_units = [PROJECTION_SIZE, NUM_CLASSES]

print(f"Image size: {IMG_SIZE} X {IMG_SIZE} = {IMG_SIZE ** 2}")
print(f"Patch size: {PATCH_SIZE} X {PATCH_SIZE} = {PATCH_SIZE ** 2} ")
print(f"Patches per image: {PATCHES}")
print(f"Elements per patch: {(PATCH_SIZE ** 2) * NUM_CHANNELS}")
print(f"Latent array shape: {LATENT_ARRAY_SIZE} X {PROJECTION_SIZE}")
print(f"Data array shape: {PATCHES} X {PROJECTION_SIZE}")

# =============== DENSE NETWORK ===============

def create_dense_block(hidden_units, dropout_rate):

    dense_layers = []
    for units in hidden_units[:-1]:
        dense_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    dense_layers.append(layers.Dense(units=hidden_units[-1]))
    dense_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(dense_layers)


# =============== PATCH CREATION AS LAYER ===============

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# =============== PATCH ENCODING LAYER ===============

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


# =============== CROSS ATTENTION MODULE ===============

def create_cross_attention_module(
    latent_dim, data_dim, projection_dim, dense_units, dropout_rate
):

    inputs = {
        # Recieve the latent array as an input of shape [1, latent_dim, projection_dim].
        "latent_array": layers.Input(shape=(latent_dim, projection_dim)),
        # Recieve the data_array (encoded image) as an input of shape [batch_size, data_dim, projection_dim].
        "data_array": layers.Input(shape=(data_dim, projection_dim)),
    }

    # Apply layer norm to the inputs
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # Create query tensor: [1, latent_dim, projection_dim].
    query = layers.Dense(units=projection_dim)(latent_array)

    # Create key tensor: [batch_size, data_dim, projection_dim].
    key = layers.Dense(units=projection_dim)(data_array)

    # Create value tensor: [batch_size, data_dim, projection_dim].
    value = layers.Dense(units=projection_dim)(data_array)

    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )
    # Skip connection 1.
    attention_output = layers.Add()([attention_output, latent_array])

    # Apply layer norm.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

    # Apply dense network.
    dense = create_dense_block(hidden_units=dense_units, dropout_rate=dropout_rate)
    outputs = dense(attention_output)

    # Skip connection 2.
    outputs = layers.Add()([outputs, attention_output])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# =============== TRANSFORMER MODULE ===============

def create_transformer_module(
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
):

    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(latent_dim, projection_dim))

    x0 = inputs
    # Create multiple layers of the Transformer block.
    for _ in range(num_transformer_blocks):

        # Apply layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)

        # Create a multi-head self-attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x0])

        # Apply layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # Apply Feedforward network.
        ffn = create_dense_block(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)

        # Skip connection 2.
        x0 = layers.Add()([x3, x2])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model


# =============== PERCEIVER MODEL ===============

class Perceiver(keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        dense_units,
        dropout_rate,
        num_iterations,
        classifier_units,
    ):
        super(Perceiver, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def build(self, input_shape):
        # Create latent array.
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )

        # Create patching module.
        self.patcher = Patches(self.patch_size)

        # Create patch encoder.
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_dim)

        # Create cross-attention module.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.dense_units,
            self.dropout_rate,
        )

        # Create Transformer module.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.dense_units,
            self.dropout_rate,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # Create a classification head.
        self.classification_head = create_dense_block(
            hidden_units=self.classifier_units, dropout_rate=self.dropout_rate
        )

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        # Augment data.
        #augmented = data_augmentation(inputs)
        # Create patches.
        patches = self.patcher(inputs)
        # Encode patches.
        encoded_patches = self.patch_encoder(patches)
        # Prepare cross-attention inputs.
        cross_attention_inputs = {
            "latent_array": tf.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }
        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.num_iterations):
            # Apply cross-attention from the latent array to the data array.
            latent_array = self.cross_attention(cross_attention_inputs)
            # Apply self-attention Transformer to the latent array.
            latent_array = self.transformer(latent_array)
            # Set the latent array of the next iteration.
            cross_attention_inputs["latent_array"] = latent_array

        # Apply global average pooling to generate a [batch_size, projection_dim] representation tensor.
        representation = self.global_average_pooling(latent_array)
        # Generate logits.
        logits = self.classification_head(representation)
        return logits


# =============== COMPILE, TRAIN AND EVALUATE MODEL ===============

def train_model(model):

    # Create LAMB optimizer with weight decay.
    optimizer = tfa.optimizers.LAMB(
        learning_rate=LEARN_RATE, weight_decay_rate=WEIGHT_DECAY,
    )

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    # fit model
    history = model.fit(
        x=x_train,#x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        #callbacks=[early_stopping, reduce_lr],
    )

    # visualise shape of model
    model.summary()

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss * 100:.2f}%")
    return history

perceiver_classifier = Perceiver(
    PATCH_SIZE,
    PATCHES,
    LATENT_ARRAY_SIZE,
    PROJECTION_SIZE,
    NUM_HEADS,
    num_transformer_blocks,
    dense_units,
    DROPOUT_RATE,
    num_iterations,
    classifier_units,
)


history = train_model(perceiver_classifier)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{OUTPUT_DIR}training_curve_epochs_{EPOCHS}.png')