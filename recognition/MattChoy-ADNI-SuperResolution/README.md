# ADNI SuperResolution Task
By Matthew Choy

> Task 5: Implement a brain MRI super-resolution network (see [Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/) [7] and a [Keras imple-
mentation](https://keras.io/examples/vision/super_resolution_sub_pixel/)) by training on the ADNI brain dataset (see Appendix for link). Create down-sampled data
(approximately by factor of 4) using either [PyTorch](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.resize.html#torchvision.transforms.functional.resize) or [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/image/resize) implementations. Network should
be trained to up-scale from 4x down-sampled input and produce a “reasonably clear image". [Normal
Difficulty]

## Table of Contents
[Table of Contents](#table-of-contents)
→ [Algorithm Description](#description-of-algorithm)
→ [How Does it Work?](#how-does-it-work)
[Inputs, Outputs and Algorithm Performance](#inputs-outputs-and-algorithm-performance)
[Implementation Details](#implementation-details)
→ [Dependencies](#dependencies)
→ [Dataset](#dataset)
→ [Data Pre-Processing](#data-pre-processing)
→ [Training, Validation and Testing Splits](#training-validation-and-testing-splits)
### Description of Algorithm
- What is this algorithm that is implemented, and what problem does it solve?
- Approximately a paragrahm

### How does it work?
- Approximately a paragraph
- Include a visualisation

## Inputs, Outputs and Algorithm Performance
- Example inputs, outputs and plots of your algorithm

## Implementation Details
### Dependencies
- Add a list of project dependencies, here, via `conda list`.

### Dataset 
- The dataset used for this task was downloaded from BlackBoard - [COMP3710/"Course Help/Resources"/ADNI MRI Dataset](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI)
### Data Pre-Processing
- The data was unzipped from the above .zip file, and moved into the project's root folder (PatternFlow/recognition/MattChoy-ADNI-SuperResolution/data/)
### Training, Validation and Testing Splits