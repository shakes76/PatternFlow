# Segmentation of Prostate 3D data set with a UNet3D.

The 3D U-Net is based on that described in the paper "3D U-Net: Learning Dense
Volumetric Segmentation from Sparse Annotation" by Cicek, Adbulkadir, Lienkamp,
Brox and Ronneberger. https://arxiv.org/abs/1606.06650. The following flow
diagram is the u-net illustrated in the paper.

![](3dunet_Cikek_etal.png)
## Semantic image segmentation of 3D images
Semantic image segmentation is the labelling of each pixel (or voxel) of an 
image with a corresponding class. For example an image of bike riders might 
have 3 classes: humans, bikes and background. Semantic segmentation is a tedious
and time-consuming task to undertake manually. Manually segmenting 3D images
requires slice by slice annotation. The goal of this u-net is to automatically
annotate 3D images after training. 
# todo add imgaes here

## Description of the algorithm

## Data download and preparation
The data is 3D MRI scans of prostate cancers. Patients had up to 8 MRI scans. 
The first, week0, was taken prior to treatment. The remaining were taken at 
week1 to week7 during a course of prostate cancer radiation therapy.

Data have been acquired as part of a retrospective MRI-alone radiation therapy 
study from the Calvary Mater Newcastle Hospital (see paper below).
Citation:  Dowling, et al. (2015), Automatic Substitute Computed Tomography 
Generation and Contouring for Magnetic Resonance Imaging (MRI)-Alone External 
Beam Radiation Therapy From Standard MRI Sequences, International Journal of 
Radiation Oncology*Biology*Physics, 93(5), pp. 1144-1153, 
https://doi.org/10.1016/j.ijrobp.2015.08.045 .

The labels are annotated with 6 classes.

            0=Background
            1=Body
            2=Bones
            3=Bladder
            4=Rectum
            5=Prostate

The data was downloaded as a doubly zipped file from the CSIRO Data access 
portal. https://data.csiro.au/collection/csiro:51392v2.

It was extracted onto a local drive with data and labels residing in respective
directories. 

        * semantic_MRs_anon 
        * semantic_labels_anon 
    
There were 211 data images and 211 matching labels. These related to 38
patients with each having from 1 to 8 scans. As multiple scans of a given 
patient were not expected to show large variations it was appropriate that 
separation into train, validation and test groupings took patients into 
consideration. That is a given patients scans could only reside in train or 
validation or test, and not span more than one group.

Segregation into train, validation and test groups was done manually, by firstly
ordering by number of scans a patient had, and then by patient ID. Clients
were then allocated into train, validation and test subdirectories on an 
approximate 70:20:10 ratio. 


![](train_val_test_allocation.png)
A review of the data structure of all images found one scan, both data and 
label, tha had a different dimension (case_019_week1) to all others. That client
had seven other scans that were expected to be similar. This file was removed. 

The data was provided in NIfTI format (nii.gz).

### training param, 


![](unet3d.png)

![](slice.png)
##  
#### and again