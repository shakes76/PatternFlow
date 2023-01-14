from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import  load_data, img_path_list


""" Load image paths """
im_path = "ISIC_image_examples"
X = img_path_list(im_path)

""" Load target data """
train_datasets = []
with open('target.txt', 'r') as f:
    train_datasets = train_datasets + f.readlines()

Y = []
for item in train_datasets:
  item = item.replace("\n", "").split(" ")
  arr = []
  for i in range(0, len(item)):
    arr.append(item[i])
  Y.append(arr)


""" initialize example image and target data for making predictions """
x, y = load_data(X, Y)


""" Initialize the model from saved_model folder """
model = keras.models.load_model('saved_model/', compile=False)

y_pred = model.predict(x)


"""Plot the four example images, their bounding boxes and classes"""
for i in range(len(y)):
    center_x = (y_pred[i][0][0][2]) * 448
    center_y = (y_pred[i][0][0][3]) * 448
    w = (y_pred[i][0][0][4]) * 448
    h = (y_pred[i][0][0][5]) * 448
    x_min = (center_x - w / 2)
    x_max = (center_x + w / 2)
    y_min = (center_y - h / 2)
    y_max = (center_y + h / 2)
    width = w
    height = h

    if (y_pred[i][0][0][0] > y_pred[i][0][0][1]):
        cls = "Not melanoma"
    else:
        cls = "Melanoma"

    im = x[i]

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.text(x_min, y_min, cls)
    plt.show()
