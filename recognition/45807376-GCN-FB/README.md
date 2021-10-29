# GCN Model Classification Using the Facebook Network Dataset

## Problem

Creating a multi-layer graph convolutional network model to conduct a semi-supervised multi-class node classification on the Facebook Large Page-Page Network.

### Facebook Large Page-Page Network

The dataset is an undirected page-page graph of Facebook sites. In which the nodes represent official Facebook pages, and the edges, the mutual likes between sites. 
The set includes pages from the categories: politicians, governmental organisations, television shows, and companies.

#### Summary of Dataset:
- Number of nodes = 22,470
- Number of edges = 171,002
- Density = 0.001
- Transitivity = 0.232


## Algorithm Description


### Dependencies

- tensorflow
- numpy
- matplotlib


## Visualisation


### Examples


## How It Works -- Justification

### Training


### Validation


### Testing Split

## References

> @misc{rozemberczki2019multiscale,
            title={Multi-scale Attributed Node Embedding},
            author={Benedek Rozemberczki and Carl Allen and Rik Sarkar},
            year={2019},
            eprint={1909.13021},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }