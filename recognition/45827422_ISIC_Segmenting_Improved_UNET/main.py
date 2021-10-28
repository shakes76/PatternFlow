import tensorflow as tf
from data_preprocess import create_generator
from improved_unet import create_model, dice_similarity
import matplotlib.pyplot as plt
import os

home = "/home/tannishpage/Documents/COMP3710_DATA"
data_folder = "ISIC2018_Task1-2_Training_Input_x2/"
gt_folder = "ISIC2018_Task1_Training_GroundTruth_x2/"

data = create_generator(os.path.join(home, data_folder),
                        os.path.join(home, gt_folder),
                        (128, 128), 750)


model = create_model((128, 128, 1))
model.summary()

train_split = int(0.7*len(data[0]))
test_split = train_split + int(0.2*len(data[0]))
val_split = test_split + int(0.1*len(data[0]))

Xtrain = data[0][0:train_split]
Ytrain = data[1][0:train_split]
Xtest = data[0][train_split:test_split]
Ytest = data[1][train_split:test_split]
Xval = data[0][test_split:val_split]
Yval = data[1][test_split:val_split]

training_results = model.fit(Xtrain, Ytrain,
                            epochs=25,
                            validation_data=(Xval, Yval),
                            batch_size=64)
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(range(len(training_results.history['loss'])),training_results.history['loss'], label='loss')
plt.plot(range(len(training_results.history['val_loss'])),training_results.history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(range(len(training_results.history['dice_similarity'])),training_results.history['dice_similarity'], label='dice_similarity')
plt.plot(range(len(training_results.history['val_dice_similarity'])),training_results.history['val_dice_similarity'], label='val_dice_similarity')
plt.legend()
plt.show()


results = model.predict(Xtest)

plt.figure(2)
plt_num = 1
for i, result in enumerate(results):
    if plt_num >= 9:
        break
    plt.subplot(3, 3, plt_num)
    plt.imshow(tf.argmax(result, axis=2), cmap='gray')
    plt_num+=1
    plt.subplot(3, 3, plt_num)
    plt.imshow(Xtest[i], cmap='gray')
    plt_num+=1
    plt.subplot(3, 3, plt_num)
    plt.imshow(tf.argmax(Ytest[i], axis=2), cmap='gray')
    plt_num+=1

plt.show()
avg_dice_coeff = 0
for i, result in enumerate(results):
    avg_dice_coeff += dice_similarity(Ytest[i], result)
print("Dice Coefficient: {}".format(avg_dice_coeff/results.shape[0]))
