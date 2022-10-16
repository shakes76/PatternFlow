from modules import create_vit_classifier
from dataset import *
import tensorflow_addons as tfa

learning_rate = 0.001
weight_decay = 0.0001
def run_final(model):
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

    final_filepath = "./best_model/checkpoint"


    model.load_weights(final_filepath)
    _, accuracy = model.evaluate(get_test_data())
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


if __name__ == "__main__":
    vit_classifier = create_vit_classifier()
    run_final(vit_classifier)
