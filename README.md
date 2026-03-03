# Official Repository for *“SpecFormer”*

This repository contains the official implementation of the paper:

> **Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios**
> *Luohe Shi, Zuchao Li, Lefei Zhang, Baoyuan Qi, Guoming Liu, Hai Zhao*
> *AAAI 2026*

---

## Overview

This codebase provides:

* Data preprocessing and rewriting pipeline
* Training framework with configurable hyperparameters
* Lightweight validation script for quick evaluation
* Support for large-scale evaluation via an external validation repository

---

## Repository Structure

```
.
├── data_rewrite/     # Scripts for dataset rewriting
├── training/         # Training scripts
├── configs/          # YAML configuration files
├── SDDE/             # Main model codes
├── validate.py       # A quick validation script, just for mocking
└── README.md
```

---

## Getting Started

### Prepare Data

First, dump the parquet files of dataset [UltraChat-200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) into:

```
./data_rewrite/raw-parquet
```

Make sure the directory structure matches the expected format required by the scripts.

---

### Rewrite Data

Use the scripts inside:

```
data\_rewrite/
```

to preprocess and rewrite the raw data into the training-ready format.

Example:

```bash
python data\_rewrite/convert.py --model-name "model name here" --model-path "path/to/model"
python data\_rewrite/concate.py --model-name "model name here" --model-path "path/to/model"
```

---

### Training

Training is performed using the scripts inside:

```
training/
```

and configuration files inside:

```
configs/
```

Example:

```bash
python training/train.py --config configs/qwen3.yaml
```

You can modify hyperparameters via the YAML config files without changing source code.

---

### Evaluation

#### 🔹 Quick Validation

For fast sanity checks or just try a few examples, use:

```bash
python validate.py
```

---


