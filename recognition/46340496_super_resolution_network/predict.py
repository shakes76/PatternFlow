import tensorflow as tf
from dataset import *
from modules import *
from train import *

total_bicubic_psnr = 0.0
total_test_psnr = 0.0

test_ds = creating_test_dataset()

rows = 3
columns = 1

model, loss, val_loss = training()

# Making Figures and Graphs
def graphs():

    # Visualise input and target
    for batch in test_ds.take(1):
        for img in batch:
            fig = plt.figure(figsize=(15, 15))
            lowres_input = get_lowres_image(img)
            prediction = predict_image(model, lowres_input)

            fig.add_subplot(rows, columns, 1)
            plt.imshow(lowres_input)
            plt.axis('off')
            plt.title("Low-res")

            fig.add_subplot(rows, columns, 2)
            plt.imshow(img)
            plt.axis('off')
            plt.title("High-res")

            fig.add_subplot(rows, columns, 3)
            plt.imshow(prediction)
            plt.axis('off')
            plt.title("Prediction")
            
            plt.show()

            upscale_lowres = tf.image.resize(lowres_input, (256, 240))

            highres_psnr = tf.image.psnr(img, upscale_lowres, max_val=255)
            prediction_psnr = tf.image.psnr(img, prediction, max_val=255)
            print("PSNR value between high-resolution and low-resolution", highres_psnr.numpy())
            print("PSNR value between high-resolution and prediciton", prediction_psnr.numpy())

    # The graph for loss and validation loss
    fig, ax = plt.subplots()
    ax.plot(loss, label='loss')
    ax.plot(val_loss, label='validation loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    ax.legend()

graphs()

