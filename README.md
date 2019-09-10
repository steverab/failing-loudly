# failing-loudly
This repository provides code, datasets, and pretrained models for our paper **"Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift"**, to be presented at Neural Information Processing Systems (NeurIPS) 2019.

Paper URL: https://arxiv.org/abs/1810.11953

## Abstract

We might hope that when faced with unexpected inputs, well-designed software systems would fire off warnings. Machine learning (ML) systems, however, which depend strongly on properties of their inputs (e.g. the i.i.d. assumption), tend to fail silently. This paper explores the problem of building ML systems that fail loudly, investigating methods for detecting dataset shift and identifying exemplars that most typify the shift. We focus on several datasets and various perturbations to both covariates and label distributions with varying magnitudes and fractions of data affected. Interestingly, we show that across the dataset shifts that we explore,  a two-sample-testing-based approach, using pre-trained classifiers for dimensionality reduction, performs best. Moreover, we demonstrate that domain-discriminating approaches tend to be helpful for characterizing shifts qualitatively and determining if they are harmful.

## Running experiments

Run all experiments using:
```
bash run_pipeline.sh
```

Run single experiments using:

```
python pipeline.py DATASET_NAME SHIFT_TYPE DIMENSIONALITY
```

Example: `python pipeline.py mnist adversarial_shift univ`

### Dependencies

We require the following dependencies:
- `keras`: https://github.com/keras-team/keras
- `tensorflow`: https://github.com/tensorflow/tensorflow
- `pytorch`: https://github.com/pytorch/pytorch
- `sklearn`: https://github.com/scikit-learn/scikit-learn
- `matplotlib`: https://github.com/matplotlib/matplotlib
- `torch-two-sample`: https://github.com/josipd/torch-two-sample
- `keras-resnet`: https://github.com/broadinstitute/keras-resnet

### Configuration

We provide shift detection using the datasets, dimensionality reduction (DR) techniques, tests, and shift types as reported in the paper. Interested users can adapt the config block in `pipeline.py` to their own needs to change:
- the DR methods used,
- how many samples to obtain from the test set,
- how many random runs should be performed,
- the significance level of the test,
- and which shifts should be simulated.

Custom shifts can be defined in `shift_applicator.py`.

## Datasets

While some datasets are already part of the Keras distribution (like MNIST, CIFAR10, and Fashion MNIST), other datasets we tested against are not directly provided. That's why we provide external datasets in the `datasets` directory for your convenience.

## Pre-trained models

This repository also provides pre-trained models for the autoencodes and BBSD for the datasets that we tested our detectors against. If you supply a dataset for which no pre-trained model is available, we will train a BBSD model for you on the fly. Convolutional autoencoder models need to be defined by you in `shift_reductor.py`, though, as we cannot ensure that all datasets reduce to the desired latent dimension and a convolutional architecture limits the way we can reduce the dimensionality.
