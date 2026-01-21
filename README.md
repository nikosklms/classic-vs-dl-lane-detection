# Lane Detection & Drivable Area Segmentation

**Traditional Computer Vision vs YOLOP**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mSE0hwnCSZ7L8kiukyOUtdBlGBHWuUJW?usp=sharing)

---

## Authors

| Name | |
|------|--|
| Vasileios Grapsopoulos | |
| Nikolaos Kalamaris | |

**Date:** January 2026

---

## Overview

This repository contains the source code used to reproduce the experiments described in our paper, comparing:

| Approach | Description |
|----------|-------------|
| **YOLOP** | State-of-the-art deep learning model for lane detection |
| **Traditional CV** | Our custom implementation using classical computer vision |

---

## How to Run

### Method 1: YOLOP (Deep Learning)

The YOLOP code is provided as a Jupyter Notebook (`.ipynb`) designed for **Google Colab** with GPU acceleration.

#### Option A: Upload Notebook (Recommended)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **"Upload"** and select `yolop.ipynb`
3. Go to `Runtime` → `Change runtime type` → Select **T4 GPU**
4. Follow the in-notebook instructions to mount Google Drive and download weights

#### Option B: Direct Link

[Open in Google Colab](https://colab.research.google.com/drive/1mSE0hwnCSZ7L8kiukyOUtdBlGBHWuUJW?usp=sharing)

> **Note:** Copy the notebook to your Drive before making changes.

---

### Method 2: Traditional CV (Our Implementation)

Our custom implementation using classical computer vision techniques.

#### Requirements

```bash
pip install opencv-python numpy
```

#### Usage

```bash
python3 ld_das.py
```

#### Configuration

Edit the input folder path in `ld_das.py`:

```python
# Change this line to use your own images:
img_dir = "images_hybrid/"  # → img_dir = "my_path/"
```

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| YOLOP | Google Account, Google Drive, GPU Runtime |
| Traditional CV | Python 3.x, OpenCV, NumPy |

---

## Project Structure

```
.
├── yolop.ipynb          # YOLOP notebook for Colab
├── ld_das.py            # Traditional CV implementation
├── images_hybrid/       # Sample input images
└── README.md
```

---

## 📄 License

This project is for educational purposes.
