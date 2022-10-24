def train():
    import matplotlib.pyplot as plt
    from keras.optimizers import Adam

    from constants import EPOCHS
    from dataset import train_data, validation_data
    from modules import UNet
    from utils import DSC, DSC_loss

    model = UNet()

    model.compile(optimizer=Adam(learning_rate=0.0003), loss=DSC_loss, metrics=[DSC])

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
    )

    plt.title("DSC Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Dice similarity coefficient")
    plt.plot(history.history["DSC"], label="Training dataset")
    plt.plot(history.history["val_DSC"], label="Validation dataset")
    plt.legend()

    return model
