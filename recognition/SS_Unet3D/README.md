# UNet3D - CSIRO 3D Male Pelvic MRI Segmentation

# Introduction
This project implements the UNet3D[[1]](#ref1) Convolutional Neural Network architecture in Tensorflow 2.6, and applies it on the (downsampled) CSIRO 3D Male Pelvic MRI dataset[[2]](#ref2) to achieve dense volumetric segmentation of the pelvis.

The model achieved a per-class Dice Similarity Co-efficient ranging from X (worst class) to Y (best class) excluding the background voxels (trained for ~Z hours on a 20GB partition of an Nvidia Tesla A100 Tensor Core GPU).

# Architecture
The UNet3D was implemented very similarly to the network described in [[1]](#ref1). Key differences include:

- Input and output shapes
- Loss Function (Unweighted Categorical Cross Entropy)
- Optimiser (ADAM)
- Augmentation regime (materialised & deterministic; not on-the-fly)

The architecture contains 19,078,662 parameters and is visualised below:

<a href="https://ibb.co/KXh9qSf"><img src="https://i.ibb.co/kxBh8vw/download-6.png" alt="download-6" border="0"></a>


# Methodology
## Data Pre-processing & Augmentation
The CSIRO dataset contains 211 anonymised 3D MRI scans (with a resolution of 256 x 256 x 128), however they belong to 38 patients only (weekly scans). Thus, there are only 38 independent samples, with the rest treated as derivatives. Although achieving good performance whilst training on small dataset is possible for this architecture, augmentation was undertaken to enrich the dataset, using the pyimgaug3d library[[3]](#ref3). Each image was augmented twice, with the transform for each augment randomly chosen between:

- Flip (Lateral)
- Elastic Deformation
- Both Flip and E.D

Each image (including the original) was also downsampled by a factor of 2 (no anti-aliasing filter was applied) to a resolution of 128 x 128 x 64.

**Note**: One image had irregular dimensions but the augmentation routine detects this and trims along the edges of the axis to conform.

The final dataset contained 633 MRI images.

## Data Splitting
Due to the independence constraints highlighted above, the train / validation / test splits were done on a patient by patient basis. The following split percentages were used, w.r.t number of patients in each split (no weighting was applied based on number of scans for each patients, which does vary):

- Train: 0.6
- Val: 0.2
- Test: 0.2

The RNG used for splitting was initialised using the same seed during training and final testing, to safeguard against data leakage and bias.

## Training
The model was trained on 30,000 batches (batch size of 1) and models saved and evaluated on the validation set every 1,000 batches. The final model was selected according to the best validation loss and test performance ascertained **subsequent to the selection**.

## Results


# Usage

Execute the steps **in the order prescribed below** to use the model.

### Limitations
- There is currently no ability to ingest parameters via a config file. Parameters are to be set in the functions directly.
- There is currently no explicit logging functionality. To interrogate the training performance, inspect the python stdout.

## 1. Augmentation - `augmenter.py`

### About
This module contains functions which pre-process and augment the CSIRO dataset. Refer to docstrings for more details.

### Environment
The augmentation script requires its own environment due to dependency conflicts.
~~~~
conda create unet3d_augmenter python=3.7
conda activate unet3d_augmenter
pip install pyimgaud3d
~~~~

### Execution
The first step is to run the augmentation script. Fill in the original data directory path (which expects the MRIs and the segmented labels in two subfolders). Specify an output path - all data (including labels) will be placed in one folder with a specific parseable naming convention which is used by the other scripts.

This script **MUST** be run even if no augmentation is desired, in order to collate, rename and downsample the data. To run without augmentation, simply set `aug_count=0` before running.

This script only needs to be run once to prepare the dataset for training.

## 2. Training & Testing - `driver.py`

### About
This module contains core functions for dataset splitting (train / val / test), training and model evaluation. Refer to docstrings for more details.

### Environment
The driver requires its own clean environment to run.
~~~~
conda create unet3d_driver python=3.7
conda activate unet3d_driver
conda install pip
pip install tensorflow nibabel matplotlib pandas
~~~~

### Execution - Train
Ensure `train()` is enabled for run and `test_saved_model()` is commented out.
Modify `train()` by filling in all parameters such as model name, augmented (cleaned) data directory, RNG seed, desired split ratios, training loop count, validation (and model save) frequency.

**IMPORTANT**: The `feature_map_scale` allows the user to easily widen or thin-out the network (to make the network deeper or shallower will involve modifying `model.py`). The 19 million parameter network trained and visualised above can be created by setting `feature_map_scale=2`. Setting `feature_map_scale=1` reduces the parameter count to 4.7 million, allowing training to occur on a Desktop GPU, such as an RTX2080.

When run, `train()` will output the loss and metrics per batch and save the model periodically (as specified by you) in the cleaned data directory, under a subdirectory.

### Execution - Test
Ensure `test_saved_model()` is enabled for run `train()` is commented out.
Modify `test_saved_model()` by pointing it to the augmented (cleaned) data directory and the path of any model saved during the training process. It will execute the DSC metric measurement on all data samples for the provided split type (e.g. test). **Ensure the training RNG seed is re-used here to avoid bias**.

# References
<a id="ref1">[1]</a>  Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016, June 21). _3D U-Net: Learning dense volumetric segmentation from sparse annotation_. arXiv.org. Retrieved October 03, 2021, from https://arxiv.org/abs/1606.06650.

<a id="ref2">[2]</a> Dowling, Jason, Greer, Peter (2021): Labelled weekly MR images of the male pelvis. v2. CSIRO. Data  Collection. https://doi.org/10.25919/45t8-p065

<a id="ref3">[3]</a> Liu, S. (n.d.). _SIYULIU0329/pyimgaug3d_. GitHub. Retrieved October 5, 2021, from https://github.com/SiyuLiu0329/pyimgaug3d.

<a id="ref4">[4]</a> Bhattiprolu, S. (2021, April 28). _215 - 3D U-net for semantic segmentation_. YouTube. Retrieved October 15, 2021, from https://www.youtube.com/watch?v=Dt73QWZQck4.
