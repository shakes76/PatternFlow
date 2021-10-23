import tensorflow_datasets as tfds
from tensorflow.keras import callbacks, losses, metrics
from tensorflow_addons import optimizers

import aoi_akoa  # register dataset
from perceiver import Perceiver
from preprocessing import preprocess


def main():
    splits, info = tfds.load(
        "aoi_akoa",
        split=["train", "validation", "test"],
        data_dir="~/tensorflow_datasets",
        with_info=True,
        as_supervised=True,
    )

    num_classes = info.features["label"].num_classes
    train, validation, test = preprocess(
        *splits, batch_size=64, num_classes=num_classes
    )

    perceiver = Perceiver(
        num_blocks=8,
        num_self_attends_per_block=6,
        num_cross_heads=1,
        num_self_attend_heads=8,
        latent_dim=512,
        latent_channels=1024,
        num_freq_bands=64,
        num_classes=num_classes,
    )

    perceiver.compile(
        optimizer=optimizers.LAMB(
            learning_rate=4e-3,
            weight_decay_rate=1e-1,
        ),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=[metrics.CategoricalAccuracy(name="accuracy")],
    )

    csv_logger = callbacks.CSVLogger(filename="./training/history.csv")
    model_checkpointer = callbacks.ModelCheckpoint(
        filepath="./training/checkpoint", save_best_only=True
    )

    history = perceiver.fit(
        x=train,
        epochs=100,
        validation_data=validation,
        callbacks=[model_checkpointer, csv_logger],
    )

    perceiver.save("./training/perceiver")
    loss, accuracy = perceiver.evaluate(test)

    print("\n", "evaluation:", {"loss": loss, "accuracy": accuracy})


if __name__ == "__main__":
    main()
