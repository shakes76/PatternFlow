import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import tensorflow_addons as tfa
import math
import data as d
import config as c

num_classes = 2
# input_shape = (260, 228, 3)
input_shape = (64, 64, 3)

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
# print("keras", y_train[0])
training_it = d.training_data_iterator()
testing_it = d.test_data_iterator()


# print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

learning_rate = 0.001
weight_decay = 0.0001
# batch_size = 64
batch_size = 4
num_epochs = 50
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 2  # Size of the patches to be extract from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
latent_dim = 256  # Size of the latent array.
projection_dim = 256  # Embedding size of each element in the data and latent arrays.
num_heads = 8  # Number of Transformer heads.
ffn_units = [
    projection_dim,
    projection_dim,
]  # Size of the Transformer Feedforward network.
num_transformer_blocks = 4
num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.
classifier_units = [
    projection_dim,
    num_classes,
]  # Size of the Feedforward network of the final classifier.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")
print(f"Latent array shape: {latent_dim} X {projection_dim}")
print(f"Data array shape: {num_patches} X {projection_dim}")

# data_augmentation = keras.Sequential(
#     [
#         # layers.Normalization(),
#         # layers.R
#         layers.Rescaling(
#             scale=1./127.5, offset=-1
#         ),

#         layers.Resizing(image_size, image_size),
#         layers.RandomFlip("horizontal"),
#         # layers.RandomZoom(
#         #     height_factor=0.2, width_factor=0.2
#         # ),
#     ],
#     name="data_augmentation",
# )
# Compute the mean and the variance of the training data for normalization.
# data_augmentation.layers[0].adapt(x_train)

B = tf.keras.backend

@tf.function(experimental_relax_shapes=True)
def gelu(x):
    return 0.5 * x * (1 + B.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

def create_ffn(hidden_units, dropout_rate):
    ffn_layers = []
    for units in hidden_units[:-1]:
        ffn_layers.append(layers.Dense(units, activation=tf.nn.gelu)) #tf.nn.gelu -> gelu

    ffn_layers.append(layers.Dense(units=hidden_units[-1]))
    ffn_layers.append(layers.Dropout(dropout_rate))

    ffn = keras.Sequential(ffn_layers)
    return ffn

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # batch_size = 
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        # patches = tf.reshape(patches, [-1, patch_dims])
        return patches

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


def create_cross_attention_module(
    latent_dim, data_dim, projection_dim, ffn_units, dropout_rate
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
    attention_output = layers.Attention(use_scale=True,dropout=0.1)( #remove dropout
        [query, key, value], return_attention_scores=False
    )
    # Skip connection 1.
    attention_output = layers.Add()([attention_output, latent_array])

    # Apply layer norm.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    # Apply Feedforward network.
    ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
    outputs = ffn(attention_output)
    # Skip connection 2.
    outputs = layers.Add()([outputs, attention_output])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

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
        ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        # Skip connection 2.
        x0 = layers.Add()([x3, x2])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model

class Perceiver(keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        ffn_units,
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
        self.ffn_units = ffn_units
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

        # Create cross-attenion module.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create Transformer module.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # Create a classification head.
        self.classification_head = create_ffn(
            hidden_units=self.classifier_units, dropout_rate=self.dropout_rate
        )

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        # Augment data.
        # augmented = data_augmentation(inputs)
        augmented = inputs
        # print("inputs vs augmented", inputs.shape, augmented.shape)
        # Create patches.
        patches = self.patcher(augmented)
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

        # Apply global average pooling to generate a [batch_size, projection_dim] repesentation tensor.
        representation = self.global_average_pooling(latent_array)
        # Generate logits.
        logits = self.classification_head(representation)
        return logits

def run_experiment(model):

    # Create LAMB optimizer with weight decay.
    optimizer = tfa.optimizers.LAMB(
        learning_rate=learning_rate, weight_decay_rate=weight_decay,
    )

    #loss function for train step
    loss_f = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=loss_f,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    #call to initialize
    model(np.zeros((1, 64, 64, 3)))

    model.load_weights("weights.h5")

    for epoch_num in range(num_epochs):
        training_it = d.training_data_iterator()
        print(f"epoch:{epoch_num}/{num_epochs}")
        total_cor = 0
        total_ce = 0
        for i in tqdm(range(c.iterations_per_epoch)):
            images, labels = training_it.next()
            ce, prediction = train_step(model, images, labels, optimizer, loss_f)
            # print(labels, prediction)
            correct_num = correct_num_batch(labels, prediction)
            print('loss: {:.4f}, accuracy: {:.4f}'.format(ce, correct_num / c.batch_size))
            total_cor += correct_num 
            total_ce += ce

        model.save_weights("weights.h5", save_format='h5')

        with open("train_history.csv", 'a') as f:
            f.write(f"{epoch_num}, {total_cor / c.train_num}, {total_ce / c.train_num}\n")

        print(f"Top-1 (train) OVERALL ACC {total_cor / c.train_num}")

        sum_correct_num = 0
        sum_loss = 0
        testing_it = d.test_data_iterator()
        for i in tqdm(range(c.test_iterations)):
            try:
                images, labels = testing_it.next()
                loss, prediction = test_step(model, images, labels, loss_f)
                # print("labels, pred", labels, prediction)
                correct_num = correct_num_batch(labels, prediction)
                print(f"correct num:{correct_num}")
                sum_correct_num += correct_num 
                sum_loss += loss
                print('loss: {:.4f}, accuracy: {:.4f}'.format(loss, correct_num / c.batch_size))
            except Exception as e:
                print(e)
                print("probs ran out, continuing")
                
            print(f"TEST {sum_correct_num / c.test_num}, {sum_loss / c.test_num}\n")


    

        with open("test_history.csv", 'a') as f:
            f.write(f"{epoch_num}, {sum_correct_num / c.test_num}, {sum_loss / c.test_num}\n")  

    # x_train, y_train = training_it.next()
    # x_test, y_test = testing_it.next()

    # print("FIT ON SHAPE")
    # print(x_train.shape)
    # print(y_train.shape)
    # print(y_train)

    # # Fit the model.
    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     # training_it,
    #     batch_size=1,
    #     epochs=1,
    #     # validation_data=testing_it,
    #     validation_split=0.1,
    #     callbacks=[early_stopping, reduce_lr],
    # )

    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # _, accuracy, top_5_accuracy = model.evaluate(testing_it)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history

perceiver_classifier = Perceiver(
    patch_size,
    num_patches,
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
    num_iterations,
    classifier_units,
)


@tf.function
def train_step(model, images, labels, optimizer, loss_f):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        # ce = cross_entropy_batch(labels, prediction, label_smoothing=c.label_smoothing)
        loss = loss_f(labels, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

@tf.function
def test_step(model, images, labels, loss_f):
    prediction = model(images, training=False)
    loss = loss_f(labels, prediction)
    return loss, prediction

def correct_num_batch(y_true, y_pred):
    # print(y_true, y_pred)
    # y_true = tf.reshape(y_true, (batch_size, 1))
    pred = tf.argmax(y_pred, -1)
    pred = tf.reshape(pred, (min(c.batch_size, pred.shape[0]), 1))
    pred = tf.cast(pred, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)
    # print(y_true, pred)
    # print(true, tf.argmax(y_pred, -1))


    # correct_num = tf.equal(y_true, tf.argmax(y_pred, -1))
    correct_num = tf.equal(y_true, pred)
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num

# def train(model, data_iterator, optimizer, log_file):

#     sum_ce = 0
#     sum_correct_num = 0

#     for i in tqdm(range(c.iterations_per_epoch)):
#         images, labels = data_iterator.next()
#         ce, prediction = train_step(model, images, labels, optimizer)
#         correct_num = correct_num_batch(labels, prediction)

#         sum_ce += ce * c.batch_size
#         sum_correct_num += correct_num
#         print('ce: {:.4f}, accuracy: {:.4f}, l2 loss: {:.4f}'.format(ce,
#                                                                      correct_num / c.batch_size,
#                                                                      l2_loss(model)))



history = run_experiment(perceiver_classifier)
