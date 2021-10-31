import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# customized functions
from methods import *
from model import *


if __name__ == "__main__":

    # only uncomment this line the first time you run the main.py
    # images with irregular shape will be resize and save to replace 
    # resize_and_replace_volumes()

    # images are splited into tarin/val/test sets on patient basis 
    target_dir = "./unet3d/dataset/semantic_MRs_anon"
    train_list, val_list, test_list = train_val_test_split(target_dir)


    # create 3d Generators for model inputs
    train_generator = DataGenerator(train_list)
    validation_generator = DataGenerator(val_list)
    test_generator = DataGenerator(test_list)
    print(len(train_generator), len(validation_generator), len(test_generator))

    # initialize an instance of 3d unet model or load  exiting Model
    model = Unet_3d()
    # model = tf.keras.models.load_model('./unet3d/results/models/model.h5', compile=False)
 
    # model configuration: try different optimizer, learning rate, loss .....
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay = decay_rate), loss="sparse_categorical_crossentropy", metrics=[dice_coef])
    model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001, decay = 0.0000001),
    loss="sparse_categorical_crossentropy",
    metrics=dice_coef,
    )
    print(model.summary())

    history = model.fit(train_generator, epochs=10, validation_data=validation_generator, shuffle=True)

    model_name = "unet_aug1"
    model.save("./unet3d/results/models/" + model_name + ".h5")

    _ = plt.plot(history.history['loss'])
    _ = plt.plot(history.history['val_loss'])
    _ = plt.title('Train & Validation Loss')
    _ = plt.ylabel('loss')
    _ = plt.xlabel('epoch')
    _ = plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.savefig('./unet3d/results/images/' + model_name + '_loss.png')


    test_results = model.evaluate(test_generator, verbose=0)
    print("############ Test loss:", test_results[0], "########### Test ")
    print(model.metrics_names)
    print(test_results)


    # compare prediction with groud truth
    # take one case from test sets
    x, y = test_generator[0]
    y_pred = model.predict(x, batch_size=1)
    print("y_pred: ", y_pred.shape)

    y_pred = tf.argmax(y_pred, axis=-1)
    print("y_pred: ", y_pred.shape)

    for i in [0, 32, 64, 127]:
    # for i in [64,]:
        # prediction
        _ = fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 6))
        _ = ax1.imshow(y[0,:,:,i])
        _ = ax1.set_title('True label')

        # True lablel
        _ = ax2.imshow(y_pred[0,:,:,i])
        _ = ax2.set_title('Predicted label')
        plt.savefig('./unet3d/results/images/' + "The " + str(i+1) + 'th slice of' + model_name + '.png')




