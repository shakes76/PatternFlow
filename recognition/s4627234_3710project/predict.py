
import os 
import matplotlib.pyplot as plt
import tensorflow as tf
import train
import dataset
import modules


def plot_data(history, type):
    plt.figure(figsiz = (10, 5))

    if type == 'acc': add = 'DSC'
    else: add = 'loss'

    plt.plot(history.history['' + add], label='Training ' + add)
    plt.plot(history.history['val_' + add], label='Validation ' + add)
    plt.title('Test vs Validation ' + add)
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('' + add)
    plt.savefig('./images/'+ add +'.png')
    plt.show()


def main():
    val_x, val_y = dataset.load_val()
    model, history = train.training(data_reshape = False)
    plot_data(history, 'acc')
    plot_data(history, 'loss')

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