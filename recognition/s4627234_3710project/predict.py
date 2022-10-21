
import os 
import matplotlib.pyplot as plt
import tensorflow as tf
import train
import dataset
import modules




def main():
    val_x, val_y = dataset.load_val()
    model = tf.keras.models.load_model('imp_unet_model.h5', 
                custom_objects={'DSC': modules.DSC, 'DSC_loss': modules.DSC_loss})    

    pred = model.predict(val_x)
    gt = tf.convert_to_tensor(val_y, dtype=tf.float32)
    print(modules.DSC(gt, pred))

    number_list = range(50) # image number (any)
    for each in number_list:
        fig = plt.figure()

        fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(val_x[each])
        ax.title.set_text("Testing Image")
        ax.axis('off')

        result=pred[each]>0.5
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(result*255, cmap="gray")
        ax.title.set_text("Predict Image")
        ax.axis('off')

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(val_y[each])
        ax.title.set_text("Ground Truth")
        ax.axis('off')

        plt.savefig('./images/output'+ str(each) +'.png')


if __name__ == "__main__":
    main()