# StyleGan_Oasis

This project is an implementation of StyleGan based on the Oasis brain dataset.

Reference: https://github.com/rosinality/style-based-gan-pytorch.git

Original Paper: https://arxiv.org/abs/1812.04948

## Dataset

The OASIS datasets hosted by central.xnat.org provide the community with open access to a significant database of neuroimaging and processed imaging data across a broad demographic, cognitive, and genetic spectrum an easily accessible platform for use in neuroimaging, clinical, and cognitive research on normal aging and cognitive decline. All data is available via [www.oasis-brains.org](https://www.oasis-brains.org/).s



## Requirements

- Python3

- Pytorch >= 1.0.0
- lmdb
- tqdm

## Usage

- ### prepare the data

  ```
  !python prepare_data.py --out LMDB_PATH --n_worker N Data_path
  ```

  This step will generate a LMDB Dataset for training

- ### Training StyleGan

