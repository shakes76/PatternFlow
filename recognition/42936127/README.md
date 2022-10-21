# Generative VQVAE on the OASIS brain dataset




## Dependencies

### Tensorflow v2
### Numpy 1.23.3
### Matplotlib 3.5.3

## Data preparation

The brain image data should have the following file structure within the root of the files

```
keras_png_slices_data/
...slices/
......keras_png_slices_train/
.........case_001_slice_0.nii.png
.........case_001_slice_1.nii.png
.........       . . . 
......keras_png_slices_test/
.........case_441_slice_0.nii.png
.........case_441_slice_1.nii.png
.........       . . . 
......keras_png_slices_validate/
.........case_402_slice_0.nii.png
.........case_402_slice_1.nii.png

...slices_seg/
......keras_png_slices_seg_train/
.........seg_001_slice_0.nii.png
.........seg_001_slice_1.nii.png
.........       . . . 
......keras_png_slices_seg_test/
.........seg_441_slice_0.nii.png
.........seg_441_slice_1.nii.png
.........       . . . 
......keras_png_slices_seg_validate/
.........seg_402_slice_0.nii.png
.........seg_402_slice_1.nii.png
.........       . . .  

```
