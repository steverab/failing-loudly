# failing-loudly
Code repository for our paper "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift": https://arxiv.org/abs/1810.11953

## Abstract

We might hope that when faced with unexpected inputs, well-designed software systems would fire off warnings. Machine learning (ML) systems, however, which depend strongly on properties of their inputs (e.g. the i.i.d. assumption), tend to fail silently. This paper explores the problem of building ML systems that fail loudly, investigating methods for detecting dataset shift and identifying exemplars that most typify the shift. We focus on several datasets and various perturbations to both covariates and label distributions with varying magnitudes and fractions of data affected. Interestingly, we show that while classifier-based methods designed to explicitly discriminate between source and target domains perform well in high-data settings, they perform poorly in low-data settings. Moreover, across the dataset shifts that we explore, a two-sample-testing-based approach using pre-trained classifiers for dimensionality reduction performs best.

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

**Note:** SVHN and MNIST-to-USPS datasets are not included as part of Keras and hence must be downloaded separately (see **Datasets and pre-trained models** below).

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

## Datasets and pre-trained models

We provide additional datasets and pre-trained models for dimensionality reduction on some of these datasets. They can be downloaded here: https://syncandshare.lrz.de/getlink/fiC9NeMN1ajbeeG41ppdAC8B/

The downloaded zip-file contains two folders: `datasets` and `saved_models`. Copying them to the root directory containing the code files will ensure that the datasets and pre-trained models are automatically detected. If you supply supply a dataset for which no pre-trained model is available, we will train BBSD model for you. Convolutional autoencoder models need to be defined by you in `shift_reductor.py`, though, as we cannot ensure that all datasets reduce to the desired latent dimension and a convolutional architecture limits the way we can reduce the dimensionality.
