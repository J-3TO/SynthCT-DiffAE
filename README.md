# Diagnostically Meaningful, Privacy Aware Synthetic CT:Class-Conditional Filtered Latent Vector Sampling for Diffusion Autoencoders

This repository implements a 3-stage pipeline for generating privacy-aware synthetic CT data
and evaluating it on a downstream classification task:

1. **Train a diffusion autoencoder** on real CT slices. 
2. **Generate synthetic data**: fit a class-conditional kernel density estimate (KDE) over the
   real latent codes, sample new codes from it, reject any candidate that lands too close to a
   real code, and decode the accepted codes into images with the
   diffusion model.
3. **Train and compare classifiers** Model for PE detection trained on real slices,
   on the synthetic slices from stage 2, and on a class-balanced mix of both.

The example pipeline here targets the [RSNA-STR Pulmonary Embolism Detection](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection)
dataset.

## Setup

```bash
pip install -r requirements.txt
```
You'll also need:
- The RSNA-STR-PE-Detection dataset from Kaggle (`train.csv` + the competition zip archive).

### Running the scripts

Adjust the respective .yml files. Then run:

```bash
python -m 0_trainDiffAE.0_p1_train
python -m 1_generateData.1p1_extract_latent_codes
python -m 1_generateData.1p2_generate_fake_dataset_RSNA_PE
```


```bash
cd 2_train_resnet
python 2p1_train_pe_classifiers.py
```
