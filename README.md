> [!IMPORTANT]  
> This repository is the official implementation of **Adaptive-SN2N**.

<br><br>
[![Github commit](https://img.shields.io/github/last-commit/WeisongZhao/SN2N)](https://github.com/WeisongZhao/SN2N/)
[![License](https://img.shields.io/badge/License-ODbL-blue.svg)](https://opendatacommons.org/licenses/odbl/1-0/)
[![DOI](https://img.shields.io/badge/DOI-10.3724%2FPXLIFE.2025--0010-blue)](https://doi.org/10.3724/PXLIFE.2025-0010)
<br>

<p>
<h1 align="center">Adaptive-SN2N</h1>
<h5 align="center">Artifact-suppressed and adaptive self-inspired learning denoising for super-resolution fluorescence microscopy.</h5>
</p>

<br>

This repository contains the official source code for our paper, "**Artifact-suppressed and adaptive self-inspired learning denoising for super-resolution fluorescence microscopy**". This work introduces **Adaptive-SN2N**, an enhanced self-inspired learning framework for image denoising in fluorescence microscopy. Our method is specifically designed to suppress background artifacts, a common challenge in biological image analysis, by incorporating a risk-aware adaptive normalization strategy and a Gaussian-weighted overlap-tile inference mechanism.

Published paper DOI: [https://doi.org/10.3724/PXLIFE.2025-0010](https://doi.org/10.3724/PXLIFE.2025-0010)

<br><br><br>

<div align="center">

✨ [**Introduction**](#-Introduction) **|**  🔧 [**Installation**](#-Installation)  **|** 🎨 [**Data Generation**](#-Data-Generation) **|**  💻 [**Training & Inference**](#-Training--Inference) **|** 📜 [**License**](#-License)

</div>

---

## ✨ Introduction

**Adaptive-SN2N** is an enhanced self-inspired learning framework for image denoising in fluorescence microscopy. It overcomes the limitations of standard self-supervised methods by:

1.  **Suppressing Background Artifacts:** Utilizing a risk-aware adaptive normalization strategy.
2.  **Adaptive Inference:** Implementing a Gaussian-weighted overlap-tile inference mechanism for seamless reconstruction.
3.  **Dual-Mode Learning:** Supporting both global and local learning modes to adapt to different noise characteristics.

## 🔧 Installation

### Dependencies 
  - Python >= 3.8
  - PyTorch >= 1.12
  - Other dependencies: `numpy`, `scipy`, `scikit-image`, `tifffile`, `matplotlib`, `pandas`

### Instruction

1. Clone the repository.

    ```bash
    git clone https://github.com/YourUsername/aSN2N.git
    cd aSN2N    
    ```

2. Install the required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

## 🎨 Data Generation

Before training the model, you need to generate an unsupervised training dataset from your raw microscopy images(preferably with images in tif format) using `Scripts_aSN2N_datagen.py`. This script generates "Global" and "Local" mode datasets required for the adaptive learning process.

### Usage

Run the script from the command line:

```bash
python Scripts_aSN2N_datagen.py --train_data_path "path/to/raw/images" --output_base_path "path/to/save/data" --both_modes
```

### Parameters

```
    -----Parameters------
    --train_data_path: (Required)
        Path to the directory containing your raw training images (e.g., .tif files).
    --output_base_path: (Optional, default: './output')
        Base path where the generated datasets will be saved. 
        Subdirectories 'global' and 'local' will be created automatically.
    --both_modes: (Optional, flag)
        If set, generates BOTH global and local datasets regardless of the adaptive decision.
    --vis_patches: (Optional, flag)
        Enable visualization of individual patches for debugging/analysis.
    --vis_overlay: (Optional, flag)
        Enable visualization of risk overlay on images.
    --export_csv: (Optional, flag)
        Export patch metrics to CSV for analysis.
```

## 💻 Training & Inference

The training and inference processes are integrated into `Scripts_aSN2N_train.py`. This script supports multi-GPU training and is configured via a JSON file.

### 1. Configuration

Create a configuration file (e.g., `Config/your_config.json`) based on the provided example in `Config/example.json`.

**Example Configuration (`Config/example.json`):**

```json
[
    {
        "dataset_name": "experiment_name",
        "train_data_path": "path/to/generated/training_data",
        "test_path": "path/to/raw/data/for/inference",
        "epochs": 60,
        "train_batch_size": 32,
        "test_batch_size": 1,
        "reg_sparse": 0,
        "reg": 0.5,
        "work_mode": "local",
        "inference_mode": "local"
    }
]
```

*   **work_mode**: Set to `"local"` or `"global"` depending on the dataset being trained.
*   **inference_mode**: Defines the inference strategy (usually matches `work_mode`).

### 2. Execution

Before running the script, ensure the `config_path` in `Scripts_aSN2N_train.py` points to your configuration file, or modify the script to load your specific JSON file.

```python
# In Scripts_aSN2N_train.py
config_path = './Config/your_config.json' 
```

Run the training script:

```bash
python Scripts_aSN2N_train.py
```

The script will:
1.  Automatically detect available GPUs.
2.  Distribute experiments defined in the JSON config across GPUs.
3.  Train the model and perform inference on the `test_path` data.
4.  Save results (images and checkpoints) in the `./images/` directory.

## 📜 License

This software and corresponding methods can only be used for **non-commercial** use, and they are under **Open Data Commons Open Database License v1.0**.

[![License](https://img.shields.io/badge/License-ODbL-blue.svg)](https://opendatacommons.org/licenses/odbl/1-0/)
