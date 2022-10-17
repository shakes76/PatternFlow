import tensorflow as tf
from dataset import *
from modules import *
from train import *

total_bicubic_psnr = 0.0
total_test_psnr = 0.0

test_ds = creating_test_dataset()

for batch in test_ds.take(1):
    for img in batch:
        lowres_input = get_lowres_image(img)
        highres_img = tf.image.resize(img, (256, 240))
        # prediction = tf.image.resize(model, (256, 240))
        lowres_img = tf.image.resize(lowres_input, (256, 240))

        # lowres_img_arr = tf.keras.preprocessing.image.img_to_array(lowres_img)
        # highres_img_arr = tf.keras.preprocessing.image.img_to_array(highres_img)
        # # predict_img_arr = tf.keras.preprocessing.image.img_to_array(prediction)
        # bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
        # test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

        # total_test_psnr += test_psnr
        # total_bicubic_psnr += bicubic_psnr
        img_plot = plt.imshow(img)
        plt.show()


        # print("PSNR of predict and high resolution is %.4f" % test_psnr)
        plot_results(lowres_img, batch, "lowres")
        plot_results(highres_img, batch, "highres")
        # plot_results(prediction, batch, "prediction")

print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))