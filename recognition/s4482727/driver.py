import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from iunet import build_iunet
from tensorflow.keras.optimizers import Adam
from dice import Dice, sorensen_dice
import matplotlib.pyplot as plt
from itertools import islice

# target size to scale input images to
IMG_WIDTH = 128
IMG_HEIGHT = 96

# For loading images from disk. Normalises the images to values between 0 and 1
input_image_generator = ImageDataGenerator(rescale=1 / 255)
# For creating generators from already loaded and pre-processed images
image_generator = ImageDataGenerator()

# number of images in training set
TRAINING_COUNT = 2594

# load training inputs from disk
training_input_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=TRAINING_COUNT,  # just get all images in one batch for now
    shuffle=False,
    classes=["ISIC2018_Task1-2_Training_Input_x2"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='input')

# load training ground truths from disk
training_gt_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=TRAINING_COUNT,  # just get all images in one batch for now
    shuffle=False,
    classes=["ISIC2018_Task1_Training_GroundTruth_x2"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode='input')

# get all loaded training input images in a single array
x_train = training_input_gen.next()[0]
# get all loaded training ground truth images in a single array
y_train = training_gt_gen.next()[0]

# number of images from training set to use as validation
VALIDATION_COUNT = 300

# split training set into validation set and new training set
x_val = x_train[0:VALIDATION_COUNT]
y_val = y_train[0:VALIDATION_COUNT]
x_train = x_train[VALIDATION_COUNT:]
y_train = y_train[VALIDATION_COUNT:]

# create training and validation generators which create pairs of the form
# (input, ground truth) in batch sizes of 32
training_gen = image_generator.flow(x=x_train, y=y_train, batch_size=32)
val_gen = image_generator.flow(x=x_val, y=y_val, batch_size=32)

# build, compile, and train improved U-Net
model = build_iunet()
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=Dice())
model.fit(x=training_gen, epochs=50, validation_data=val_gen)

# save in case we want to use the model later
model.save("model.tf")

# number of images in test set
TEST_COUNT = 100

# load test inputs from disk
test_input_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=TEST_COUNT,
    shuffle=False,
    classes=["ISIC2018_Task1-2_Validation_Input"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='input')

# load test ground truths from disk
test_gt_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=TEST_COUNT,
    shuffle=False,
    classes=["ISIC2018_Task1_Validation_GroundTruth"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode='input')

# get all loaded test input images in a single array
x_test = test_input_gen.next()[0]
# get all loaded test ground truth images in a single array
y_test = test_gt_gen.next()[0]

# use generator to get test pairs of the form (input, ground truth)
test_gen = image_generator.flow(x_test, y_test, shuffle=True)

# use model to segment test images
predictions = model.predict(x_test)

# number of tests to plot
TESTS = 3
# plot rows of the form (input, ground truth, prediction)
figure, cells = plt.subplots(TESTS, 3)
for i in range(TESTS):
    inputt, truth = next(islice(test_gen, i, None))
    prediction = model.predict(inputt)
    dsc = sorensen_dice(truth, prediction)
    cells[i, 0].title.set_text('Input {0}'.format(i))
    cells[i, 0].imshow(inputt[0])
    cells[i, 1].title.set_text('Ground Truth {0}'.format(i))
    cells[i, 1].imshow(truth[0], cmap='gray')
    cells[i, 2].title.set_text('Prediction {0} (dsc={1})'
                               .format(i, "%.4f" % dsc))
    cells[i, 2].imshow(prediction[0], cmap='gray')
figure.tight_layout()
plt.axis('off')
plt.show()
