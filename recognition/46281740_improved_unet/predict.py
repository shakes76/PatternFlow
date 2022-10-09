from glob import glob
import dataset as dp
import modules as iu
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# set the path
dataset_path = "D:/Data/Mylecture/COMP3710/lab_report/data/"

seg_test_path = sorted(glob(dataset_path + "ISIC-2017_Test_v2_Part1_GroundTruth/*.png"))
test_path = sorted(glob(dataset_path + "ISIC-2017_Test_v2_Data/*_???????.jpg"))

# create the dataset
test_ds = tf.data.Dataset.from_tensor_slices((test_path, seg_test_path))

# load the dataset
test_ds = test_ds.map(dp.process_image)

test, seg_test = next(iter(test_ds.batch(len(test_path))))

# load the model file
modelpath = './model/mymodel.h5'
model = tf.keras.models.load_model(modelpath,
                                   custom_objects={"dice_coefficient_avg": iu.dice_coefficient_avg,
                                                   "dice_coefficient": iu.dice_coefficient, "dice_loss": iu.dice_loss})

# compute DSC
prediction = model.predict(test)
tf.print("Average DSC for all labels: ", iu.dice_coefficient_avg(seg_test, prediction))
tf.print("DSC for each label: ", iu.dice_coefficient(seg_test, prediction))


# plot the predict results
def display(title_list, image_list, cmap='viridis'):
    fig, ax = plt.subplots(1, len(title_list), figsize=(10, 10))
    for j, k in enumerate(title_list):
        ax[j].set_title(k)
    for j, k in enumerate(image_list):
        ax[j].imshow(k, cmap=cmap)
    return plt


random_images = [random.randint(1, len(test)) for i in range(3)]

for i in random_images:
    plt = display(['Input Image', 'True Segmentation', 'Predicted Segmentation'],
                  [test[i][:, :, 0], tf.argmax(seg_test[i], axis=-1), tf.argmax(prediction[i], axis=-1)], cmap='gray')
    plt.savefig('./image/prediction_' + str(i) + ".png")
    plt.show()