import tensorflow as tf
from tensorflow.keras import layers
from cross_attention import cross_attention_layer
from transformer import transformer_layer
from dense_net import dense_block
from fourier_encode import FourierEncode
import tensorflow_addons as tfa

class Perceiver(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        data_size,
        latent_size,
        proj_size,
        num_heads,
        num_transformer_blocks,
        dense_layers,
        num_iterations,
        classifier_units,
        max_freq, 
        num_bands,
        lr,
        epoch,
        weight_decay
    ):
        super(Perceiver, self).__init__()

        self.latent_size = latent_size
        self.data_size = data_size
        self.patch_size = patch_size
        self.proj_size = proj_size
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_layers = dense_layers
        self.iterations = num_iterations
        self.classifier_units = classifier_units
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.lr = lr
        self.epoch = epoch
        self.weight_decay = weight_decay

         # Create latent array.
        self.latent_array = self.add_weight(
            shape=(self.latent_size, self.proj_size),
            initializer="random_normal",
            trainable=True,
        )

        # Create patching module.
        # self.patcher = Patches(self.patch_size)

        # Create patch encoder.
        self.fourier_encoder = FourierEncode(self.max_freq, self.num_bands)

        # Create cross-attenion module.
        self.cross_attention = cross_attention_layer(
            self.latent_size,
            self.data_size,
            self.proj_size,
            self.dense_layers,
        )

        # Create Transformer module.
        self.transformer = transformer_layer(
            self.latent_size,
            self.proj_size,
            self.num_heads,
            self.num_transformer_blocks,
            self.dense_layers,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # Create a classification head.
        self.classify = dense_block(self.classifier_units)

    # def build(self, input_shape):
       
    #     super(Perceiver, self).build((32,260,228,1))

    def call(self, inputs):
        # Augment data.
        # augmented = data_augmentation(inputs)
        # Create patches.
        # patches = self.patcher(augmented)
        # Encode patches.
        encoded_imgs = self.fourier_encoder(inputs)

        cross_attention_inputs = [
            tf.expand_dims(self.latent_array, 0),
            encoded_imgs
        ]
        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.iterations):
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

class ModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt, ckpt_manager):
        self.ckpt_manager = ckpt_manager
        self.ckpt = ckpt

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt.start_epoch.assign_add(1)
        self.ckpt_manager.save()

## trainning
def train(model, train_set, val_set, test_set):
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    optimizer = tfa.optimizers.LAMB(
        learning_rate=model.lr, weight_decay_rate=model.weight_decay,
    )

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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

    # epoch_end = ModelCallback(checkpoint, ckpt_manager)
    # Fit the model.
    history = model.fit(
        X_train, y_train,
        epochs=5,
        callbacks=[early_stopping, reduce_lr],
        validation_data=(X_val, y_val),
        batch_size=32,
        validation_batch_size=32
    )

    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    ## plot stuff here

    # Return history to plot learning curves.
    return history


