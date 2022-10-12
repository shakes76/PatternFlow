"""
Assumptions:

Steps / Key Functions:
1. Instantiate model
2. Select Optimzer
3. Compile model
4. Create Checkpoint callback
5. Train the model


References:
1) https://keras.io/examples/vision/image_classification_with_vision_transformer/
2) https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093

"""


def main():
    from tensorflow import keras
    import tensorflow_addons as tfa
    from modules import vit_classifier
    from dataset import import_data


    ##############################  HYPERPARAMETERS  ###################################

    # data
    BATCH_SIZE = 128
    IMAGE_SIZE = 196  # We'll resize input images to this size
    NUM_CLASS = 2
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)

    # patches
    PATCH_SIZE = 14  
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

    # transformer-econder
    PROJECTION_DIM = 768
    NUM_HEADS = 12
    NUM_ENCODER_LAYERS = 12

    # mlp head
    MLP_HEAD_UNITS = [3072]  # Size of the dense layers of the final classifier

    # model
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 200
    DROPOUTS = {"mha": 0.2, "encoder_mlp": 0.2, "mlp_head": 0.5}



    ##############################   IMPORT DATA  ###################################

    data_train, data_validate, data_test = import_data(IMAGE_SIZE, BATCH_SIZE)

    ##############################  TRAINING SCRIPT  ###################################
    # Run Experiment --> Instantiate model, Select optimzer, compile, checkpoint, train and evaluate

    # instantiate model
    vit_classifier = vit_classifier()
    print(vit_classifier.summary())

    # select optimzer
    optimizer = tfa.optimizers.AdamW(
        LEARNING_RATE=LEARNING_RATE, WEIGHT_DECAY=WEIGHT_DECAY
    )
    #     optimizer = tf.optimizers.Adam(LEARNING_RATE=LEARNING_RATE)


    # compile
    vit_classifier.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    # create checkpoint callback
    checkpoint_filepath = "C:\\Users\\lovet\\Documents\\COMP3710\\Report\\adni\\checkpoint\\"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    # train the model
    history = vit_classifier.fit(
        x=data_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=data_validate,
        callbacks=[checkpoint_callback],
    )


    # evaluate the model 
    vit_classifier.load_weights(checkpoint_filepath)
    _, accuracy, = vit_classifier.evaluate(x=data_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


if __name__ == "__main__":
    main()
