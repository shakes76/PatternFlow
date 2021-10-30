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
I have cloned Datknet library from https://github.com/AlexeyAB/darknet .
In order to build the yoloV3 model, 
Also, I have enabled CUDA, CUDNN and OPEN-CV library to accelerate the model with GPU computing.

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
So I wrote a python script to transform those images into the required format of input labels.

The darknet also requires us to set some parameters for the network.
Here is the configuration of my yoloV3 network:


# Step3 training yoloV3 model：

After setting up all the configurations, we can use the following command 




The trained weights is stored in [ url  ]


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
<img src="https://github.com/slowlier/PatternFlow/blob/87d7c88fc9b019cf00e1db6313465c398bd28092/recognition/S4607867_Jiaqiyu/pics/my_map_calc.png" width=100% height=100%>
</p>
</n>
<p align="center">
  The mAP result over 0.8 IOU threshold
</p>
<p align="center">
<img src="https://github.com/slowlier/PatternFlow/blob/937f496c7b24bc01819e35d079f288e19fb9f234/recognition/S4607867_Jiaqiyu/pics/my_map_calc@0.8.png" width=100% height=100%>
</p>

# Conclusion














