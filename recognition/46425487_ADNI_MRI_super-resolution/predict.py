import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

# import the model and test data set created by train.py
model = keras.models.load_model('./trained_model')
test_ds = tf.data.Dataset.load('./test_data')

total_resize_psnr = 0.0
total_predic_psnr = 0.0

for batch in test_ds.take(1):
    # get the low_res and high_res images from test dataset
    for i, low_res in enumerate(batch[0]):
        high_res = batch[1][i]

        # create a reconstructed image from the low_res image using the model  
        inp = np.expand_dims(low_res, axis=0)
        predicted = model.predict(inp)
        predicted = predicted[0]
        
        # resize low res image to match the others
        low_res = keras.preprocessing.image.array_to_img(low_res)
        low_res = low_res.resize((256, 256))
        low_res = keras.preprocessing.image.img_to_array(low_res)
        low_res = low_res / 255.0

        resize_psnr = tf.image.psnr(low_res, high_res, max_val=1)
        predic_psnr = tf.image.psnr(predicted, high_res, max_val=1)

        total_resize_psnr += resize_psnr
        total_predic_psnr += predic_psnr


        print(f"PSNR of low res and high res for image {i} is {resize_psnr:.4f}")
        print(f"PSNR of predicted and high res for image {i} is {predic_psnr:.4f}")

print(f"Average PSNR of low res and high res for images is {total_resize_psnr/32:.4f}")
print(f"Average PSNR of predicted and high res for images is {total_predic_psnr/32:.4f}")

