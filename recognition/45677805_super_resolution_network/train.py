from dataset import *
from modules import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

#load the train validation and test dataset
train_data = get_train()
val_data = get_validation()
test_data = get_test()

#resize the train , validation and test data set and prefetc the data into CPU
resized_train_data = resize_img_flow(train_data, 32)
resized_train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
resized_val = resize_img_flow(val_data, 32)
resized_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
resized_test = resize_img_flow(test_data, 32)
resized_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# get the srcnn model
model = get_model()

# loss function
loss_fun = MeanSquaredError()
optimizer_Adam = Adam(0.001)
# mse_metrics = 


# # by change base log law log10(x) = ln(x)/ln(10)
# def log10fun(x):
#     numerator = tf.math.log(x)
#     deno = tf.math.log(x)
#     return numerator/deno

# # signal noise ratio
# def psnr_metrics(crop, original):
#     mse = tf.losses.mean_squared_error(crop, original)
#     psnr = 10 * log10fun(1/tf.sqrt(mse))
#     return psnr


model.compile(loss=loss_fun, optimizer=optimizer_Adam, metrics=['MeanSquaredError'])

model.fit(resized_train_data, epochs=1, validation_data=resized_val)