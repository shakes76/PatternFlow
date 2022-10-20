from dataset import get_datasets
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np

from train import get_lowres_image, plot_results, train, upscale_image

def predict():
    upscale_factor = 4
    model, test_ds = train(1)
    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0
    for batch in test_ds.take(1):
        for index, img in enumerate(batch):
            lowres_img = get_lowres_image(img, 4)
            w = lowres_img.shape[0] * upscale_factor
            h = lowres_img.shape[1] * upscale_factor
            highres_img = tf.image.resize(img, (w,h))
            prediction = upscale_image(model, lowres_img)
            lowres_img = tf.image.resize(lowres_img, (w, h))
            lowres_img_arr = img_to_array(lowres_img)
            highres_img_arr = img_to_array(highres_img)
            predict_img_arr = img_to_array(prediction)
            
            bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val = 255)
            plot_results(lowres_img, index, "lowres")
            plot_results(highres_img, index, "highres")
            plot_results(prediction, index, "prediction")
            test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val = 255)

            total_bicubic_psnr += bicubic_psnr
            total_test_psnr += test_psnr
            print(
                "PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
            )
            print("PSNR of predict and high resolution is %.4f" % test_psnr)
            
    
    print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 8))
    print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 8))

def main():
    predict()

if __name__ == "__main__":
    main()