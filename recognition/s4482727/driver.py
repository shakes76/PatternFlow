import os
from itertools import islice
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from iunet import build_iunet
from tensorflow.keras.optimizers import Adam
from dice import Dice
import matplotlib.pyplot as plt

image_generator = ImageDataGenerator()
input_image_generator = ImageDataGenerator(rescale=1/255)

training_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=2594,
    shuffle=False,
    classes=["ISIC2018_Task1-2_Training_Input_x2"],
    target_size=(128, 96),
    class_mode='input')

training_gt_gen = input_image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=2594,
    shuffle=False,
    classes=["ISIC2018_Task1_Training_GroundTruth_x2"],
    target_size=(128, 96),
    color_mode="grayscale",
    class_mode='input')

x_train = training_gen.next()[0]
y_train = training_gt_gen.next()[0]
print(x_train.shape, y_train.shape)

data_gen = image_generator.flow(x=x_train, y=y_train)

model = build_iunet()
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=Dice())
model.fit(x=data_gen, epochs=50)

model.save("model.tf")

figure, cells = plt.subplots(3, 3)
for i in range(3):
    inputt, truth = next(islice(data_gen, i, None))
    prediction = model.predict(inputt)
    cells[i, 0].imshow(inputt[0])
    cells[i, 1].imshow(truth[0], cmap='gray')
    cells[i, 2].imshow(prediction[0], cmap='gray')
plt.axis('off')
plt.show()
