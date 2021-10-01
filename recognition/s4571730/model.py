import tensorflow as tf
from tensorflow.keras import layers
from cross_attention import cross_attention_layer
from transformer import transformer_layer
from dense_net import dense_block
from fourier_encode import FourierEncode
import tensorflow_addons as tfa


def augment():
    # TODO
    pass

# class Patches(layers.Layer):
#     def __init__(self, patch_size):
#         super(Patches, self).__init__()
#         self.patch_size = patch_size

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         patches = tf.image.extract_patches(
#             images=images,
#             sizes=[1, self.patch_size, self.patch_size, 1],
#             strides=[1, self.patch_size, self.patch_size, 1],
#             rates=[1, 1, 1, 1],
#             padding="VALID",
#         )
#         patch_dims = patches.shape[-1]
#         patches = tf.reshape(patches, [batch_size, -1, patch_dims])
#         return patches

class Perceiver(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        dense_layers,
        num_iterations,
        classifier_units,
        max_freq, 
        num_bands
    ):
        super(Perceiver, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_layers = dense_layers
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units
        self.max_freq = max_freq
        self.num_bands = num_bands

    def build(self, input_shape):
        # Create latent array.
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )

        # Create patching module.
        # self.patcher = Patches(self.patch_size)

        # Create patch encoder.
        self.fourier_encoder = FourierEncode(input_shape, self.max_freq, self.num_bands)

        # Create cross-attenion module.
        self.cross_attention = cross_attention_layer(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.dense_layers,
        )

        # Create Transformer module.
        self.transformer = transformer_layer(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.dense_layers,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # Create a classification head.
        self.classify = dense_block(
            hidden_units=self.classifier_units
        )

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = self.patcher(augmented)
        # Encode patches.
        encoded_patches = self.fourier_encoder(patches)

        # Prepare cross-attention inputs.
        cross_attention_inputs = [
            tf.expand_dims(self.latent_array, 0),
            encoded_patches
        ]

        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.num_iterations):
            latent_array = self.cross_attention(cross_attention_inputs)
            latent_array = self.transformer(latent_array)
            cross_attention_inputs[0] = latent_array

        # Apply global average pooling
        outputs = self.global_average_pooling(latent_array)

        # Generate logits.
        logits = self.classify(outputs)
        return logits

def data_augmentation():
    pass


## trainning
def train(model, train_set, val_set, test_set, lr=0.004, weight_decay=0.0001, num_epoch=10):

    optimizer = tfa.optimizers.LAMB(
        learning_rate=lr, weight_decay_rate=weight_decay,
    )

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            # tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    # Create a learning rate scheduler callback.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    # Fit the model.
    history = model.fit(
        x=train_set,
        validation_data=val_set,
        epochs=num_epoch,
        callbacks=[early_stopping, reduce_lr],
    )

    _, accuracy = model.evaluate(test_set)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history


