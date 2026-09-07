# MedFairXNet — Melanoma Detection Benchmark

Explainable and uncertainty-aware deep learning for melanoma detection: a six-model benchmark (DenseNet121, ResNet101, EfficientNetV2, ConvNeXt-Tiny, Swin-Tiny, and the proposed MedFairXNet) evaluated on HAM10000, with exploratory external testing on PH².

## Overview

MedFairXNet is implemented as an EfficientNetV2B0 backbone with a Convolutional Block Attention Module (CBAM) and Monte Carlo (MC) Dropout for predictive-uncertainty estimation. This repository contains the training, evaluation, and analysis code supporting the accompanying manuscript.

## Datasets

- **HAM10000** — primary training/validation/test dataset (10,015 dermoscopic images). Not redistributed here; download from the official ISIC Archive / Harvard Dataverse source.
- **PH²** — external test subset. Not redistributed here; download from the official PH² source.

Place downloaded images according to the structure noted in `check_datasets.py`.

## Repository Structure

```
├── train_medfairxnet_expA.py                        # MedFairXNet training script
├── generate_medfairxnet_validation_predictions.py   # Validation-set prediction generation
├── check_datasets.py                                # Dataset presence/integrity checks
├── check_roi_csv.py                                 # ROI/annotation CSV validation
├── inspect_isic2018.py                              # ISIC 2018 metadata inspection
├── inspect_isic_structure.py                         # ISIC folder-structure inspection
├── inspect_ph2.py / inspect_ph2_excel.py             # PH² metadata inspection
├── medfairxnet_training_code.txt                     # Reference training-configuration notes
├── scripts/                                          # Supporting scripts
├── requirements.txt                                  # Python dependencies
└── results/                                          # Predictions, metrics, and figures (add before submission)
```

## Installation

```bash
git clone https://github.com/Packiyalakshmikiruba/MedFairXNet-Melanoma.git
cd MedFairXNet-Melanoma
pip install -r requirements.txt
```

## Reproducing Results

1. Download HAM10000 and PH² from their official sources (see Datasets above).
2. Run `check_datasets.py` to verify dataset structure and integrity.
3. Run `train_medfairxnet_expA.py` to train MedFairXNet.
4. Run `generate_medfairxnet_validation_predictions.py` to generate validation/test predictions.
5. Prediction and metric files are written to `results/`.

## Citation

If you use this code, please cite the accompanying manuscript (details to be added upon publication).

## License

Released under the MIT License (see `LICENSE`).

## Contact

A. Packiyalakshmi — dronaisuvidhafoundation@gmail.com
