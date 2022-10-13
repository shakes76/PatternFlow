import tensorflow as tf
from dataset import *
from modules import *
from train import *

total_bicubic_psnr = 0.0
total_test_psnr = 0.0

test_img = creating_test_dataset()
model, callbacks, loss_fn, optimizer, checkpoint_filepath = model_checkpoint()


for index, test_img_path in enumerate(test_img[50:60]):
    img = tf.keras.preprocessing.image.load_img(test_img_path)
    lowres_input = get_lowres_image(img, UPSCALE_FACTOR)
    w = lowres_input.size[0] * UPSCALE_FACTOR
    h = lowres_input.size[1] * UPSCALE_FACTOR
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = tf.keras.preprocessing.image.img_to_array(lowres_img)
    highres_img_arr = tf.keras.preprocessing.image.img_to_array(highres_img)
    predict_img_arr = tf.keras.preprocessing.image.img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print(
        "PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
    )
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    plot_results(lowres_img, index, "lowres")
    plot_results(highres_img, index, "highres")
    plot_results(prediction, index, "prediction")

print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))