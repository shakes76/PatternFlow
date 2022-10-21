<<<<<<< HEAD
# Detecting melanoma on ISIC dataset using YOLO model
### The problem and the dataset
The International Skin Imaging Collaboration (ISIC) dataset is a popular medical image dataset which is widely used by researchers for image analysis. These research and analysis contribute to the fight against a common type of cancer – melanoma, which takes lives of more than 37000 every year. Recent developments in computer vision and neural networks can help in early diagnosis of melanoma. One of these algorithms is You Only Look Once – YOLO. 
### YOLO algorithm
YOLO is the algorithm for real-time object detection. It solves both prediction of bounding boxes and classification tasks at once, which used to be solved separately. Moreover, the algorithm can make the predictions for all the objects after it looked at image at once – that’s it’s called “You Only Look Once”. In my project I used YOLOv1 – the first of many versions of this algorithm.
The basic idea behind YOLO is to split an image into a grid of dimension SxS, like in the picture below:

![image](https://user-images.githubusercontent.com/22009116/197185230-cebe94f1-70f6-45f0-b289-0ff71ca6111e.png)

The grid cell, where the center of an object is located is responsible for predicting this object class and its bounding box. In YOLOv1 default model, two bounding boxes are predicted for each cell, and while the model is training, bad bounding boxes are removed, and only one most appropriate is left. The final output of the model are vectors, predicted for each cell with values: [C, X, Y, W, H, R], where 
* C – is the class of the object
* X, Y – are center coordinates of the bounding box
* W, H – are width and height of the bounding box, and
* R – is the response variable, which shows the probability of that center of an object is in that cell 
The final result after training should look like this:

![image](https://user-images.githubusercontent.com/22009116/197189215-ef989737-ec9a-4d72-af6e-de5e056a20d7.png)

### How my code works
#### Data preprocessing
For my project I used ISIC dataset’s image set, segmentation masks, and class labels ground truth in .csv file. I faced some problems with loading ISIC test and validation sets, so I divided ISIC training data with 2000 instances into training, validation, and test sets with the proportion of 80:10:10 percent.
ISIC dataset doesn’t have annotations of bounding box coordinates, so I had to derive them using segmented mask images - data.py file contains code for that. Then I derived class labels, and concatenated them with bounding box coordinates, so that it became a vector of int values [Xmin, Ymin, Xmax, Ymax, Cls] for each image. The data in my project has two class labels: 0 – if an image doesn’t contain a picture of melanoma, and 1 – if it does. I wrote all these vectors to target.txt file to make it more convenient to read target data. In train.py file I load target data using target.txt file, instead of calculating bounding box coordinates again. 
#### Building the model
I built the model following an official paper of YOLOv1, which looks like the picture below:

![image](https://user-images.githubusercontent.com/22009116/197186090-3d234575-7452-4adf-87ee-4ef2418a4a1f.png)

The difference of that scheme from my model, is that my model has the output of shape 7x7x12, because I had 2 classes, and 2 bounding boxes predicted for each grid cell. The shape must be: SxSxC+(2*B)
#### Loss function
YOLO has a complex loss function, which consists of three parts: bounding box coordinates loss, confidence of having an object loss and classification loss.

![image](https://user-images.githubusercontent.com/22009116/197186226-f1fb4d8e-cad4-4e1c-bb56-b1885b9726b7.png)

#### Learning rate scheduler
To prevent the model from converging in bad local minima, I wrote learning rate scheduler function, which reduces the learning rate after some number of epochs. I was able to train only 30 epochs due to Google Colab limitations, and the learning rate is first 0.1, from epoch#10 it’s 0.01, and from epoch#20 it’s 0.001. 
You can find all the model components, loss function, and learning rate scheduler in modules.py file
#### Running prediction with saved model
After training I saved the whole model into saved_model directory to be able to perform object detection without training the model again. The code for initializing saved model and predicting examples is located in predict.py file. You can run it to see how what results the model gives
### Results
Unfortunately, I didn’t get satisfying results by this time. The first problem is that training loss fluctuated, and the validation loss did not change during training, and consequently, I got bad prediction results. After analyzing my code, I came to a decision, that these mistakes are due to bad loss function. Hopefully, I will be able to deal with these mistakes in my further work.


![image](https://user-images.githubusercontent.com/22009116/197187749-5bed0065-c458-40c2-8c21-595b50bf073f.png)
![image](https://user-images.githubusercontent.com/22009116/197187900-5a2c61b7-12c3-4dcc-a558-84aadd6909c6.png)
![image](https://user-images.githubusercontent.com/22009116/197188024-05781b0f-8fb5-4005-acaa-6cbb4eeded23.png)


