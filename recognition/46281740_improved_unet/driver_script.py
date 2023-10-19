from glob import glob
import dataset as dp
import modules as iu
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset_path = "D:/Data/Mylecture/COMP3710/lab_report/data/"

seg_test_path = sorted(glob(dataset_path + "ISIC-2017_Test_v2_Part1_GroundTruth/*.png"))
seg_train_path = sorted(glob(dataset_path + "ISIC-2017_Training_Part1_GroundTruth/*.png"))
seg_val_path = sorted(glob(dataset_path + "ISIC-2017_Validation_Part1_GroundTruth/*.png"))
test_path = sorted(glob(dataset_path + "ISIC-2017_Test_v2_Data/*_???????.jpg"))
train_path = sorted(glob(dataset_path + "ISIC-2017_Training_Data/*_???????.jpg"))
val_path = sorted(glob(dataset_path + "ISIC-2017_Validation_Data/*_???????.jpg"))

# create the dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_path, seg_train_path))
val_ds = tf.data.Dataset.from_tensor_slices((val_path, seg_val_path))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, seg_test_path))

# load the dataset
train_ds = train_ds.map(dp.process_image)
val_ds = val_ds.map(dp.process_image)
test_ds = test_ds.map(dp.process_image)

test, seg_test = next(iter(test_ds.batch(len(test_path))))

# create the UNet model and start straining
model = iu.unet()
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy', metrics=[iu.dice_coefficient_avg])

history = model.fit(train_ds.batch(10),
                    epochs=8,
                    validation_data=val_ds.batch(10))

# save the model as .h5 file
tf.keras.models.save_model(model, './model/mymodel.h5')

# plot the DSC of train and validate dataset
plt.figure(figsize=(8, 5))
plt.title("Dice Similarity Coefficient")
plt.plot(history.history["dice_coefficient_avg"], label="Training DSC")
plt.plot(history.history["val_dice_coefficient_avg"], label="Validation DSC")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('./image/Dice.png')
plt.show()

# compute the DSC of each prediction image
prediction = model.predict(test)
tf.print("Average DSC for all labels: ", iu.dice_coefficient_avg(seg_test, prediction))
tf.print("DSC for each label: ", iu.dice_coefficient(seg_test, prediction))