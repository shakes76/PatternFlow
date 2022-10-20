import dataset
import modules
from tensorflow.keras.optimizers import Adam


def training(data_reshape = False):

    train_x, train_y, test_x, test_y = dataset.load_dataset(data_reshape)
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    model = modules.UNet_imp()

    # fit the model with normal learning rate
    model.compile(optimizer = Adam(0.0005), loss = modules.DSC_loss, metrics=['accuracy', modules.DSC])

    history = model.fit(train_x, train_y,  validation_data= (test_x, test_y),
                            batch_size=8,shuffle='True',epochs=10)

    # # decrease the learning rate to do the further learning if need
    # model.compile(optimizer = Adam(learning_rate = 1e-4), loss = modules.DSC_loss, metrics = [modules.DSC]) 
    # history2 = model.fit(train_x, train_y,  validation_data= (test_x, test_y),
    #                         batch_size=8,shuffle='True',epochs=10)

    return model, history