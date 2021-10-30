# Improved UNet

Yiyun Zhang 45133507

## Description

This is the implementation of ISIC data segmentation using an improved U-Net.

Improved UNet Network Architecture:
![Improved UNet Architecture](./images/Improved_UNet.png)

## Instructions
1. Download the ISIC2018 datasets at: https://challenge.isic-archive.com/data/#2018, specifically these two folders:
    - ISIC2018_Task1-2_Training_Input
    - ISIC2018_Task1_Training_GroundTruth
2. Put these two folders under `./datasets`
3. Open the terminal and `cd` to the project directory (`/recognition/Yiyun_ISIC`)
4. Install the Python libraries listed in `./requirements.txt`: `pip install -r ./requirements.txt`
5. Run the driver script `python3 ./main.py`

## Results

![Predictions Plot](./images/plot_sample.png)

![Metrics Plot](./images/plot_metrics_sample.png)

## Troubleshoot
`tensorflow.python.framework.errors_impl.InvalidArgumentError: buffer_size must be greater than zero. [Op:ShuffleDatasetV3]`
- The datasets have not been properly read.
- Check if the `ISIC_DIR` directory in `main.py` and `ISIC_INPUT_DIR`/`ISIC_GROUNDTRUTH_DIR` folder names in `data.py` match.

## References
\[1\] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. \[Online\].
Available: https://arxiv.org/abs/1802.10508v1
