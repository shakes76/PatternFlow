import tensorflow as tf
import dice
import data_processing
import plots
import Unet_model

def main():
    train_ds,val_ds,test_ds = data_processing.data_processing()

    model = Unet_model.Unet()

    model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics=[dice.dice_coefficient] )

    # model.summary()

    plots.plots(val_ds,model)


if __name__ == "__main__":
    main()