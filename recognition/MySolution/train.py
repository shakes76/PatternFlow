import stellargraph as sg
from keras.callbacks import EarlyStopping


def handle_training(module):
    model = module.get_model()
    train_gen = module.get_train_gen()
    val_gen = module.get_val_gen()
    es_callback = EarlyStopping(
        monitor="val_acc",
        patience=50,
        restore_best_weights=True
    )
    history = model.fit(
        train_gen,
        epochs=200,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[es_callback],
    )
    sg.utils.plot_history(history)


