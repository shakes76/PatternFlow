# Skin Lesion Segmentation on the ISIC Challenge 2017 Dataset Using Improved U-NET

**Avatar Azka - 47286238**

Submission for the final report in the course COMP3710 (Pattern Recognition & Analysis)

This is an implementation of the Improved UNET model, as detailed in the paper by Isensee et al. (2018) [1], applied to segment images of skin lesions on the ISIC challenge 2017 dataset.

## Algorithm Overview
![IUNET Architecture](static/IUNET.png)
The Improved UNET model is split into a context pathway and a localization pathway. The context pathway performs feature extraction, the result of which is then taken as the input for the localization pathway which, as the name suggests, aims to localize particular areas of interest. Both pathways are connected to each other via skip connections, which allows the network to recover finer details to be included in the final output mask.

Beyond the diagram, Improved UNET utilizes dropout layers between convolutions in each context module. It also uses instance normalization in place of batch normalization, which the author owes to its small batch sizes destabilizing batch normalization.

### How It Works
The model outputs images of a similar shape as its input image. To generate a mask from this output image, the implemented algorithm isolates the first RGB channel of the image, then sets whatever elements are of a value greater than or equal to a given threshold value (in this case, 0.5) to 1.0, and whatever is lower than that is set to 0.0. This resulting mask can then be compared with the ground truth of the data.

### Loss and Metrics
The performance of this model was measured using the Sørensen–Dice coefficient [2].

### Dataset
When training the model, the algorithm loads 800 training elements at a time, though it shuffles the loader per epoch, effectively covering the majority of if not entire dataset over 15 total epochs. Due to time and resource constraints, only 100 of the 150 total elements of the validation dataset are used to measure the validation metrics during training.

## Prerequisites
### Folder Structure
The dataset module itself is structure-agnostic, though train.py and predict.py expect a folder structure as detailed below:
```
data/
├─ test/
│  ├─ data/
│  ├─ truth/
├─ training/
│  ├─ data
│  ├─ truth
├─ validation/
│  ├─ data
│  ├─ truth
```

### Dependencies
```
matplotlib == 3.6.1
numpy == 1.23.4
pandas == 1.5.1
torch == 1.12.1
torchvision == 0.13.1
cudatoolkit == 11.3.1
```

## Reproducibility of Results
Due to the shuffling of the dataset during training, it cannot be guaranteed that the resulting model after training can be exactly reproduced over multiple iterations.



## Notes
The current hyperparameters are, admittedly, suboptimal. However, due to resource and time constraints, the batch size is set to 1 as going any higher leads to out-of-memory errors on my machine. This has been tested on the Rangpur compute system, and it seems to be able to tolerate higher batch sizes of at least 10, though execution times were faster locally.

## References
1. (Isensee et al. (2018))[https://arxiv.org/abs/1802.10508v1]
2. (Wikipedia: Sørensen–Dice coefficient)[https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient]
