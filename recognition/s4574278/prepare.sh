#!/bin/bash

# VAR
 STORE=/root/autodl-nas
 WORKSPACE=/root/autodl-tmp

# PREPARE workspace
 unzip $STORE/ISIC2018_Task1-2_Training_Data_uq.zip -d $STORE/
 cp $STORE/ISIC2018_Task1_Training_GroundTruth_x2 $WORKSPACE/ground_truth
 cp $STORE/ISIC2018_Task1-2_Training_Input_x2 $WORKSPACEp/JPEGImages
 mkdir $WORKSPACE/Annotations


# Extract annotation and do preprocessing
#jupyter nbconvert --execute --to data-result.ipynb preprocess.ipynb


conda install -c conda-forge opencv nvidia-apex -y

pip install torch-summary -y