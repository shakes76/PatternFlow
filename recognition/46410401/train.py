from modules import create_vit_classifier
from dataset import *
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    # Set checkpoint weights that save the best validation accuracy model
    checkpoint_filepath = "./tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        get_train_data(),
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=get_valid_data(),
        callbacks=[checkpoint_callback],
    )

    # Plot training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(get_test_data())
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history


if __name__ == "__main__":
    setup_folders()
    vit_classifier = create_vit_classifier()
    final_history = run_experiment(vit_classifier)
