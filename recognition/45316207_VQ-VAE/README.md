# Synthetic Brain MRI Image Generation with VQ-VAE

## COMP3710 Pattern Flow Report

Alex Nicholson (45316207)

---

Details:

* Model: [VQ-VAE (original)](https://arxiv.org/abs/1711.00937)
* Dataset: [OASIS](https://www.oasis-brains.org/#data)

Goals:

* “Reasonably clear image”
* [Structured Similarity (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity) index of over 0.6

---

## TODO

- [x] Data importing
- [x] Model class
- [x] Model basic training function
- [x] Live training performance data logging
- [ ] Output image results visualisation
- [ ] Implement saving of output images to file (ideally get images that show the progress throughout training too - can adapt from my lab2 plotting function)
- [ ] Training metrics over time plot
- [ ] SSIM performance calculation
- [ ] Port my code over to the hpc for speedy slurm training
- [ ] Do a big training run to push the standrard of output generations
- [ ] Tune hyperparameters until results meet the standard
- [ ] Report writeup, etc.


---

## Documentation (WIP)

### Setup

1. Install the conda envirinoment...
2. Download the OASIS dataset from [this link](https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download)
3. Activate the conda environment
4. Run train.py
