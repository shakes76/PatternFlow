# Yolo V3 Object Detection （based on AlexyAB darknet）

# Task introduction：
The task I choose for COMP3710 report is Yolo Object Detection task.
It is widely used algorithm in computer vision tasks such as image annotation,vehicle counting, activity recognition,or face detection.
The task here is to identify the Lesion Objects on patients' skin pictures. 
Because the downsampled ISIC-Challenge-2018 dataset has no test set (Also, there is only test set without ground truth labels on the challenge official website),The dataset I used in this assignment was ISIC-Challenge-2017 dataset, which can be downloaded from https://challenge.isic-archive.com/data/  
</n>
It includes a training subset of 2000 samples, a validation subset of 150 samples and a test subset of 600 samples.
Each of the samples contains a lesion image in JPEG format and a corresponding superpixel mask in PNG format.
</n>
E.G. :
</n>
<p align="center">
  Image of a Lesion Sample
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/156879a9709865f3b9ea5639f414373dd9c72cdc/recognition/S4607867_Jiaqiyu/darknet/data/ISIC_example/ISIC_0000001.jpg" width=50% height=50%>
</p>

</n>


<p align="center">
  Image of a Corresponding Superpixel Mask
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/156879a9709865f3b9ea5639f414373dd9c72cdc/recognition/S4607867_Jiaqiyu/darknet/data/ISIC_example/ISIC_0000001_segmentation.png" width=50% height=50%>
</p>



# Step1 building environment：

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
<p align="center">
  The network Structure of Darknet
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/e90a19a3986fb5c8888bae01fe3714dc2014098e/recognition/S4607867_Jiaqiyu/pics/darknet_structure.png" width=50% height=50%>
</p>



I have cloned Datknet library from https://github.com/AlexeyAB/darknet (An imporved version over the original darknet).
Also, I have enabled CUDA, CUDNN and OPEN-CV library to accelerate the model with GPU computing.
</n>
I am using Windows10 as OS, so I followed AlexeyAB's tutorial - compile on Windows part.
I attempted to use Cmake GUI to compile darknet, but failed.
Then, I decided to compile darknet using vcpkg. This is the vcpkg compiling recommended by AlexeyAB:

1. Install Visual Studio 2017 or 2019. In case you need to download it, please go here: Visual Studio Community. Remember to install English language pack, this is mandatory for vcpkg!
2. Install CUDA enabling VS Integration during installation.
3. Open Powershell (Start -> All programs -> Windows Powershell) and type these commands:
```
Set-ExecutionPolicy unrestricted -Scope CurrentUser -Force
git clone https://github.com/AlexeyAB/darknet
cd darknet
.\build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN
```

# Step2 preparing support files： 
I have downloaded ISIC challenge 2017 as the target dataset.
It contains 2000 train samples, 150 validation samples, and 600 test samples.
The ground truth of lesion objects are stored as segmentation shadows. 
So I wrote a couple python script to generate the label files of required input format.
(see these scripts in addons folder：   https://github.com/slowlier/PatternFlow/blob/a65194361f54c4ca6f8034dce7b842cca5a5f811/recognition/S4607867_Jiaqiyu/addons/labellingISIC.py，
https://github.com/slowlier/PatternFlow/blob/a65194361f54c4ca6f8034dce7b842cca5a5f811/recognition/S4607867_Jiaqiyu/addons/trainlist.py)



The darknet also requires us to set some parameters for the network.
Here is the configuration of my yoloV3 network:

<p align="center">
  The Configuration of My YoloV3
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/a1b43ca1568d70fd927ecb452ee780c38f8f9992/recognition/S4607867_Jiaqiyu/pics/configs.png" width=80% height=80%>
</p>



# Step3 training yoloV3 model：

After setting up all the configurations, we can use the following command to train a yoloV3 model with darknet:
```
./darknet detector train cfg/ISIC_train.data cfg/yolo-ISIC.cfg 
```
If you have a pre-trained weights checkpoint in the backup, you can continue training from it with the command:
```
./darknet detector train cfg/ISIC_train.data cfg/yolo-ISIC.cfg backup/yolo-ISIC_last.weights
```


A graph showing the learning loss curve would be saved in the darknet root folder as:
<p align="center">
  The Loss Curve of Training
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/8b0c1be2d82c3c35ce1291aef389e0af23c89da0/recognition/S4607867_Jiaqiyu/darknet/backup/attempt4/chart.png" width=70% height=70%>
</p>



The final trained weights is stored in https://drive.google.com/file/d/1USyV2NJbwAmRXiiyX2i_G7kVBbDgttol/view?usp=sharing
Weights of my previous attempts can also be found at https://drive.google.com/drive/folders/1j0sUTh0i1Ou-p7pKiCKBQwzX-CilY-Va?usp=sharing

# Step4 validation on test set:

Darknet has a build-in validation command, which can returns the average IOU ratio and other evaluation metrics.
By editting the map_calc.cmd file in the darknet root folder: 

```
rem # How to calculate mAP (mean average precision)
rem darknet.exe detector map cfg/ISIC_train.data cfg/yolo-ISIC.cfg backup/yolo-ISIC_last.weight -iou_thresh 0.8
darknet.exe detector map cfg/ISIC_train.data cfg/yolo-ISIC.cfg backup/yolo-ISIC_last.weights -iou_thresh 0.8
pause
```
A mAP result on the test set can be calculated.
</n>
</n>
<p align="center">
  The mAP result over 0.5 IOU threshold
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/79c7f7f3feb69617ad62c6809e62e5e7703b1c5a/recognition/S4607867_Jiaqiyu/pics/my_map_calc.png" width=100% height=100%>
</p>
</n>
<p align="center">
  The mAP result over 0.8 IOU threshold
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/79c7f7f3feb69617ad62c6809e62e5e7703b1c5a/recognition/S4607867_Jiaqiyu/pics/my_map_calc@0.8.png" width=100% height=100%>
</p>

# Conclusion

The mAP result on 0.8 IOU threshold seems not very ideal. Maybe it is because there are some samples in my dataset which are not so typical to get identified. After all, I chose a bigger dataset than the down-sampled one on BB.   
Also, as I am running darknet on my own device, its GPU memory can only support (batch=72; subdivision=24), but can not support (batch=64, subdivision=16) .  But the training normally takes more than 10 hours, it has a high probability to lose connection if I remotely running the darknet on UQ's lab computer.












