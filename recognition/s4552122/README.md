# Segment the ISICs data set with the Improved UNet
Image segmentation of ISICs data set implemented for tensorflow

## Description

ISICs data set concludes thousands of Skin Lesion images. This recognition algorithm aims to automatically do Lesion Segmentation through an improved unet model [[1]](#References). The evaluation of segmentation has a [Dice similarity coefficient](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient) of 0.84 on the test set. 

Here is a exmaple of the original Skin Lesion image and its Ground Truth segmentation image.

​                                 <img src="images/ISIC_0000000.jpg" alt="ISIC_0000000" style="zoom:50%;" />                                    <img src="images/ISIC_0000000_segmentation.png" alt="ISIC_0000000_segmentation" style="zoom:50%;" />  

<center>
  Figure 1. Left is original Skin Lesion image and the right is its ground truth image.
</center>



















## References

[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1

