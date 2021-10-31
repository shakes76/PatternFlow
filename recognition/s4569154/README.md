README (Runqi Lin, 45691548)
====

Background
-------

Detect lesions within the ISICs data set with a YOLO network such as the original YOLO with all detections having a minimum Intersection Over Union of 0.8 on the test set and a suitable accuracy for classification.

ISIC 2018 challenge data for skin cancer - This is part of the ISIC 2018 challenge and comes with segmentation labels. The preprocessed version of this data set can be found on the course Blackboard site.

The aim of this project is to detect the lesioned skin in the image using the yolov3 target detection algorithm.

Yolov3:Real-Time Object Detection
-------

### YOLOv3 structure

YOLOv3 is extremely fast and accurate. In mAP measured at .5 IOU YOLOv3 is on par with Focal Loss but about 4x faster. Moreover, you can easily tradeoff between speed and accuracy simply by changing the size of the model, no retraining required!

The main improvements of YOLO3 are: adjusting the network structure; using multi-scale features for object detection; and replacing softmax with logistic for object classification.

![image](https://user-images.githubusercontent.com/75237235/139569918-ce4ababe-f0b3-4274-9dc3-cb48f270c424.png)

The entire structure of Yolo v3, excluding the pooling and fully connected layers, is a Darknet-53 network, and the Yolo prediction branches are fully convolutional.

As defined by the DarknetConv2D_BN_Leaky function in yolo3.model, Darknet's convolutional layer is followed by BatchNormalization (BN) and LeakyReLU. With the exception of the last convolutional layer, BN and LeakyReLU are already inseparable parts of the convolutional layer in yolo v3. together they form the minimal component.

The backbone network uses five resn structures. n represents a number, res1, res2, ... ,res8, etc., indicating that the res_block contains n res_units, which are the large components of Yolo v3. Moving up from darknet-19 in Yolo v2 to darknet-53 in Yolo v3, the former has no residual structure. yolo v3 starts to borrow the residual structure from ResNet, and using this structure allows for a deeper network structure. An explanation of the res_block can be visualised in the bottom right corner of Figure 1.1, whose basic component is also the DBL.

There is a tensor splicing (concat) operation on the prediction branch. This is implemented by concatenating the intermediate layer of darknet and the upsampling of a layer after the intermediate layer. It is worth noting that tensor concatenation is not the same as the add operation of the Res_unit structure, as tensor concatenation expands the dimensionality of the tensor, whereas add just adds directly without changing the dimensionality of the tensor.

In the overall analysis at the code level, Yolo_body has 252 layers. 23 Res_unit layers correspond to 23 add layers. 72 BN layers and 72 LeakyReLU layers, which are represented in the network structure as follows: each BN layer is followed by a LeakyReLU layer. 2 upsampling and 2 tensor splicing operations each, and 5 zero-fills correspond to 5 res_block. There are 75 convolutional layers, 72 of which are followed by a DBL composed of BatchNormalization and LeakyReLU. Three different scales of output correspond to three convolutional layers, and the final convolutional layer has 255 convolutional kernels for the COCO dataset of class 80: 3 × (80 + 4 + 1) = 255, where 3 means A grid cell contains 3 bounding boxes, 4 indicates the 4 coordinates of the box, and 1 indicates the confidence level.

### Darknet-53 feature extraction network

darknet-53 adopts the idea of resnet and adds a residual module to the original network, which is helpful to solve the gradient problem of the deep network, and each residual module consists of two convolutional layers and a shortcut connections,
Compared with yolov1 and v2, v3 has no pooling layer and no full connection layer. The down-sampling of the network is achieved by setting the stride of the convolution to 2, and the size of the image is reduced to half after each convolutional layer. The implementation of each convolutional layer consists of convolution + BN + Leaky relu , and each residual module is followed by a zero padding, as shown in the figure below.

![image](https://user-images.githubusercontent.com/75237235/139426179-403ad5cd-491b-42e4-9ab1-731950adb6df.png)

In contrast to YoloI and v2, in Yolov3 there is only a convolution layer and the size of the output feature map is controlled by adjusting the convolution step. So there is no special restriction on the input image size.

Yolov3 draws on the pyramidal feature map idea, where small feature maps are used to detect large objects, medium feature maps to detect medium objects, and large feature maps to detect small objects. The output dimension of the feature map is N*N*[3*( 4+1+80)], N*N is the number of output feature map grid points, there are 3 Anchor boxes, each box has 4 dimensions of prediction box values X,Y,W,H, where X,Y are the coordinates of the centre point of the prediction box, W,H are the width and height of the prediction box, 1 dimension of confidence of the prediction box, and 80 dimensions of object categories. So the output dimension of the first layer of feature map is 8*8*225.

Yolov3 outputs a total of 3 feature maps, the first feature map is downsampled 32 times, the second feature map is downsampled 16 times and the third is downsampled 8 times. The input image is passed through Darknet-53 (no fully-connected layer) and then through Yoloblock to generate the feature maps, which are used in two ways: first, to generate feature map one after 3*3 convolution and 1*1 convolution, and second, to generate feature map two after 1*1 convolution plus a sampling layer, which is stitched with the output of the intermediate layer of the Darnet-53 network. The same cycle is followed by feature map three.

The difference between the concat operation and the add-and-sum operation: the add-and-sum operation is derived from the ResNet idea of adding the input feature map to the corresponding dimension of the output feature map, while the concat operation is derived from the DenseNet network design idea of stitching the feature map directly according to the channel dimension, for example, an 8*8*16 feature map is stitched with an 8*8*16 feature map to produce an 8*8*32 feature map. 8*32 feature maps.

Upsample: The role of the upsample layer is to generate a large image from a small feature map by interpolation and other methods. The upsample layer does not change the number of channels in the feature map.

Datasets
-------
ISIC2018_Task1-2_Training_Data
ISIC2016_Task1-2_Test_Input

To test the model I trained, I also downloaded the corresponding test dataset containing 300 images.
https://challenge.isic-archive.com/data/

![image](https://user-images.githubusercontent.com/75237235/139570210-38c671e2-f376-4ea9-80fb-cf65e03a79e2.png)

As shown above, the dataset contains images of skin lesions. 

In a deep learning target detection task, the model is first trained using a training set. How good the training dataset is determines the upper limit of the task.
So we need to manually label the dataset first, and I used the LabelImg image target detection labeling tool.
It comes from this project: https://github.com/tzutalin/labelImg

The application interface is shown below

![1635661030](https://user-images.githubusercontent.com/75237235/139570721-95e49a99-eefd-4ed7-affd-f6bed5c56aad.jpg)

LabelImg is a graphical image annotation tool.

In addition, I also tagged the data via opencv to identify the ISIC2018_Task1_Training_GroundTruth_x2 dataset, which is more accurate than the first manual tagging.

![image](https://user-images.githubusercontent.com/75237235/139570977-40b67c38-a0b4-4797-b434-8c43085c72f2.png)

The tagged dataset generates a txt file with the corresponding name, which contains the class name, the centroid x coordinate, the centroid y coordinate, the width of the tagged box, and the height of the tagged box. This is shown in the figure below.

![image](https://user-images.githubusercontent.com/75237235/139570969-69eadf08-9266-426d-b009-923b96d41d5d.png)

