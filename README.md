# 🎨 Sketch Classification Challenge


---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction & Submission](#prediction--submission)
- [Files Reference](#files-reference)
- [Scoring](#scoring)
- [Tips for Improvement](#tips-for-improvement)

---

## Overview

This project provides a complete end-to-end pipeline for classifying 28×28 grayscale hand-drawn sketches into **10 categories**. It includes:

- **Automatic dataset download** and extraction
- **Custom PyTorch Dataset** with data augmentation
- **CNN-based model** architecture optimized for small grayscale images
- **Training loop** with validation and best-model checkpointing
- **Prediction script** with built-in API submission support

### Challenge Details

| Property          | Value                                      |
|-------------------|--------------------------------------------|
| Image size        | 28 × 28 pixels, grayscale (0–255)          |
| Training set      | 20,000 labelled images (2,000 per class)   |
| Validation set    | ~4,000 labelled images (~400 per class)    |
| Test set          | 1,000 unlabelled images                    |
| Classes           | 10 categories (defined in `classes.txt`)   |
| Metric            | Macro-averaged F1 Score × 15               |
| Max submissions   | 20                                         |

---

## Project Structure

```
d:\work\
├── README.md                     # This file
├── sketch_classification.ipynb   # Original challenge notebook
├── dataset.py                    # Dataset loading & transforms
├── model.py                      # CNN architecture
├── train.py                      # Training script
├── predict.py                    # Inference & API submission
├── best_model.pth                # Saved model weights (after training)
└── sketch_clf/                   # Dataset folder (auto-downloaded)
    ├── classes.txt               # Class name → label ID mapping
    ├── train_labels.csv          # Training labels
    ├── validation_labels.csv     # Validation labels
    ├── train/                    # 20,000 training images
    ├── validation/               # ~4,000 validation images
    └── test/                     # 1,000 unlabelled test images
```

---

## Prerequisites

- **Python 3.10+** (tested with 3.13)
- **pip** (Python package manager)
- Internet connection (for dataset download and API submission)

---

## Installation

### Step 1: Clone or navigate to the project directory

```powershell
cd d:\work
```

### Step 2: Install all required dependencies

```powershell
pip install torch torchvision scikit-learn pandas tqdm requests pillow
```

> **Note:** If you have an NVIDIA GPU with CUDA, install the CUDA version of PyTorch for significantly faster training:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 3: Verify installation

```powershell
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## Dataset

The dataset is **automatically downloaded** when you run `train.py` or `dataset.py` for the first time. No manual steps are needed.

### Manual download (optional)

If you want to download the dataset manually:

```powershell
python dataset.py
```

This will:
1. Download `sketch_clf_dataset.zip` from the challenge server
2. Extract it into the `sketch_clf/` directory
3. Print `"Dataset extracted successfully."` when done

### Class Labels

The 10 sketch categories are stored in `sketch_clf/classes.txt`. Each line corresponds to a label ID (0–9). **Do not re-sort them.**

---

## Training

### Basic Training (default: 15 epochs)

```powershell
python train.py
```

### Custom Training with Arguments

```powershell
python train.py --epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE> --lr <LEARNING_RATE>
```

| Argument       | Default | Description                              |
|----------------|---------|------------------------------------------|
| `--epochs`     | `15`    | Number of training epochs                |
| `--batch_size` | `64`    | Batch size for training and validation   |
| `--lr`         | `0.001` | Learning rate for the AdamW optimizer    |

### Examples

```powershell
# Train for 20 epochs with a smaller learning rate
python train.py --epochs 20 --lr 0.0005

# Train with a larger batch size
python train.py --epochs 10 --batch_size 128
```

### What Happens During Training

1. **Dataset download** — automatically downloads if not already present
2. **Data augmentation** — random horizontal flip and rotation (±10°) on training images
3. **Training loop** — trains the CNN with CrossEntropy loss and AdamW optimizer
4. **Validation** — evaluates macro-averaged F1 score on the validation set after each epoch
5. **Checkpointing** — saves the model weights (`best_model.pth`) whenever a new best validation F1 is achieved

### Expected Output

```
Dataset already present.
Training on cpu...
Epoch 1/4: 100%|██████████| 313/313 [00:45<00:00]
Epoch 1/4 - Train Loss: 1.3023 - Val Loss: 0.9424 - Val Macro-F1: 0.6564
--> Saved new best model with Val Macro-F1: 0.6564
Epoch 2/4: 100%|██████████| 313/313 [00:44<00:00]
Epoch 2/4 - Train Loss: 0.9736 - Val Loss: 0.8594 - Val Macro-F1: 0.6709
--> Saved new best model with Val Macro-F1: 0.6709
...
```

> **Tip:** Training on CPU takes ~45 seconds per epoch. With a CUDA GPU, this drops to ~5 seconds per epoch.

---

## Prediction & Submission

### Step 1: Generate Predictions (without submitting)

```powershell
python predict.py
```

This will:
1. Load the best model weights from `best_model.pth`
2. Run inference on all 1,000 test images
3. Print the first 10 predictions for verification

### Step 2: Submit Predictions via API

```powershell
python predict.py --submit
```

When using `--submit`, the script will:
1. Generate all 1,000 predictions
2. Prompt you for your **API Key** (from the contest webpage)
3. POST the predictions to the AI Olympiad submission endpoint
4. Print the submission result

### Command Arguments

| Argument         | Default            | Description                           |
|------------------|--------------------|---------------------------------------|
| `--model_path`   | `best_model.pth`   | Path to the saved model weights       |
| `--submit`       | `False`            | If set, submits predictions to the API|

### Example Submission Output

```
Generating predictions for 1000 test images...
Generated 1000 predictions.
Top 10 Predictions: [5, 8, 8, 4, 8, 4, 4, 7, 5, 4]
Please enter your API Key: <YOUR_API_KEY_HERE>
Submitting to API...
Submission Result:
{'status': 'SUCCESS', 'message': 'Answer for challenge maio2026_sketch_classification submitted successfully...'}
```

> ⚠️ **Important:** You are allowed a maximum of **20 API submissions**. Use them wisely! Always validate your model locally first using the validation F1 score.

---

## Files Reference

### `dataset.py`

| Function / Class    | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| `download_and_extract_data()` | Downloads and extracts the dataset ZIP                  |
| `SketchDataset`     | PyTorch Dataset class for train, valid, and test splits           |
| `get_transforms()`  | Returns train (with augmentation) and val/test transforms         |

### `model.py`

| Class       | Description                                                            |
|-------------|------------------------------------------------------------------------|
| `SketchCNN` | 3-layer CNN with BatchNorm, MaxPool, Dropout, and 2 FC layers          |

**Architecture Summary:**
```
Input (1, 28, 28)
  → Conv2d(1→32) + BN + ReLU + MaxPool → (32, 14, 14)
  → Conv2d(32→64) + BN + ReLU + MaxPool → (64, 7, 7)
  → Conv2d(64→128) + BN + ReLU + MaxPool → (128, 3, 3)
  → Flatten → FC(1152→256) + ReLU + Dropout(0.5)
  → FC(256→10) → Output
```

### `train.py`

| Function       | Description                                              |
|----------------|----------------------------------------------------------|
| `train_model()` | Full training loop with validation and checkpointing    |

### `predict.py`

| Function             | Description                                         |
|----------------------|-----------------------------------------------------|
| `predict_test_set()` | Loads model, runs inference, optionally submits      |
| `make_payload()`     | Formats predictions as API-compatible JSON           |
| `post_answer()`      | POSTs predictions to the AI Olympiad endpoint        |

---

## Scoring

Submissions are evaluated using **macro-averaged F1 score** across all 10 classes, scaled to **15 points**:

$$\text{score} = F_1^{\text{macro}} \times 15$$

| Validation F1 | Estimated Score |
|---------------|-----------------|
| 0.60          | 9.0 / 15        |
| 0.70          | 10.5 / 15       |
| 0.80          | 12.0 / 15       |
| 0.90          | 13.5 / 15       |
| 1.00          | 15.0 / 15       |

---

## Tips for Improvement

Here are some strategies to boost your F1 score beyond the baseline:

### Data Augmentation
- Add **random affine transforms** (scaling, translation)
- Use **random erasing** to simulate occlusions
- Experiment with **elastic distortion** for sketch-like deformations

### Model Enhancements
- Try a **deeper CNN** or a **ResNet-18** (modify the first convolution for 1-channel input)
- Add a **learning rate scheduler** (e.g., `CosineAnnealingLR` or `StepLR`)
- Use **label smoothing** in the CrossEntropy loss

### Training Strategies
- Train for **more epochs** (20–50) with early stopping
- Use a **lower learning rate** (e.g., `0.0003`) with more epochs
- Try **SGD with momentum** instead of AdamW
- Combine the training and validation sets for final training (after selecting best hyperparameters)

### Ensemble Methods
- Train **multiple models** with different random seeds
- Use **majority voting** or **averaged softmax probabilities** for final predictions

---

## Quick Start Summary

```powershell
# 1. Install dependencies
pip install torch torchvision scikit-learn pandas tqdm requests pillow

# 2. Train the model
python train.py --epochs 15

# 3. Generate and review predictions
python predict.py

# 4. Submit to the API (when ready)
python predict.py --submit
```

---

**Good luck with the AI Olympiad! 🚀**
