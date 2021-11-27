import tensorflow as tf
import dice
import data_processing
import plots
import Unet_model

def main():
    #Spliting the data
    train_ds,val_ds,test_ds = data_processing.data_processing()

    model = Unet_model.Unet()

    #Compile the model
    model.compile(optimizer="adam",loss = "binary_crossentropy",metrics=[dice.dice_coefficient] )

    # model.summary()

    #plots.plots(val_ds,model,num=3)

    #Training the model
    history = model.fit(train_ds.batch(32), epochs=5, validation_data=val_ds.batch(32))

    # PLot the prediction and compare with inputs and ground-truth images.
    plots.plots(test_ds, model, num=5)

    test_size = len(test_ds)

    #Compute the average dice coefficient on test data.
    dice.average_dice(test_ds, model, test_size)


if __name__ == "__main__":
    main()