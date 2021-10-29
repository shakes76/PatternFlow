import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from iunet import build_iunet
from tensorflow.keras.optimizers import Adam
from dice import Dice
import matplotlib.pyplot as plt
from itertools import islice

IMG_WIDTH = 128
IMG_HEIGHT = 96

image_generator = ImageDataGenerator()
input_image_generator = ImageDataGenerator(rescale=1/255)

training_input_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=2594,
    shuffle=False,
    classes=["ISIC2018_Task1-2_Training_Input_x2"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='input')

training_gt_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=2594,
    shuffle=False,
    classes=["ISIC2018_Task1_Training_GroundTruth_x2"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode='input')

x_train = training_input_gen.next()[0]
y_train = training_gt_gen.next()[0]
x_val = x_train[0:300]
y_val = y_train[0:300]
x_train = x_train[300:]
y_train = y_train[300:]
training_gen = image_generator.flow(x=x_train, y=y_train)
val_gen = image_generator.flow(x=x_val, y=y_val)

model = build_iunet()
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=Dice())
model.fit(x=training_gen, epochs=50, validation_data=val_gen)

model.save("model.tf")

test_input_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=100,
    shuffle=False,
    classes=["ISIC2018_Task1-2_Validation_Input"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='input')

test_gt_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=100,
    shuffle=False,
    classes=["ISIC2018_Task1_Validation_GroundTruth"],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode='input')

x_test = test_input_gen.next()[0]
y_test = test_gt_gen.next()[0]
test_gen = image_generator.flow(x_test, y_test, shuffle=True)
predictions = model.predict(x_test)

TESTS = 3
figure, cells = plt.subplots(TESTS, 3)
for i in range(TESTS):
    inputt, truth = next(islice(test_gen, i, None))
    prediction = model.predict(inputt)
    cells[i, 0].imshow(inputt[0])
    cells[i, 1].imshow(truth[0], cmap='gray')
    cells[i, 2].imshow(prediction[0], cmap='gray')
plt.axis('off')
plt.show()
