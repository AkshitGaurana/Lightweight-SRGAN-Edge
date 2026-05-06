# 🔬 Lightweight SRGAN for Edge Deployment

> **Research Project** — SRM Institute of Science and Technology · 2025  
> *Single Image Super-Resolution Through Lightweight SRGAN for Edge Deployment*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Overview

This project implements and compares two SRGAN generator configurations for 4× single-image super-resolution:

| Model | Residual Blocks | FLOPs (relative) | Target |
|---|---|---|---|
| **SRGAN-Full** | 16 | 2× | High-quality SR |
| **SRGAN-Lite** *(proposed)* | 8 | **1×** | Edge deployment |
| Bicubic | — | baseline | Classical interpolation |

The core contribution is demonstrating that halving the residual block count preserves perceptual quality while making the generator viable on constrained hardware (Raspberry Pi, Jetson Nano, mobile NPUs).

---

## 🚀 Live Demo

Run locally:

```bash
# 1. Create and activate environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Features

- 📤 **Custom image upload** — drag-and-drop any JPG/PNG
- 🔄 **3-way comparison** — LR input · Bicubic · SRGAN-Lite · SRGAN-Full
- 📊 **Live PSNR / SSIM metrics** using standard SR benchmark protocol
- ⏱ **Inference time** measured per method
- 💾 **Download** any SR result as JPEG
- 🖥 **Auto GPU/CPU** — runs on CUDA if available, falls back to CPU

---

## Project Structure

```
├── app.py          # Streamlit web app (main entry point)
├── backend.py      # SR inference pipeline (SRGAN + bicubic fallback)
├── metrics.py      # PSNR / SSIM computation
├── model.py        # SRResNet & DiscriminatorForVGG architectures
├── patch_docx.py   # Injects computed metrics into research paper DOCX
├── requirements.txt
└── results/
    └── pretrained_models/   # Place SRGAN_x4-ImageNet.pth.tar here
```

---

## Pretrained Weights

The model requires `results/pretrained_models/SRGAN_x4-ImageNet.pth.tar`.  
Without weights, SRGAN outputs fall back to **bicubic upscaling** (always clean).

Download from the [original SRGAN-PyTorch repo](https://github.com/Lornatang/SRGAN-PyTorch).

---

## Architecture

The SRResNet generator:
```
Input (LR)
  → Conv 9×9 + PReLU
  → N × ResidualBlock (Conv-BN-PReLU-Conv-BN + skip)   ← N=8 (lite) or N=16 (full)
  → Conv 3×3 + BN
  → 2 × PixelShuffle (×2 each = ×4 total)
  → Conv 9×9
Output (HR, 4× upscaled)
```

---

## Benchmark Methodology

Standard SR evaluation protocol:
1. Input image treated as **HR ground truth**
2. Downscaled 4× (bicubic) → simulated **LR input**
3. Each method reconstructs HR from LR
4. PSNR / SSIM computed vs original HR

---

## References

1. Ledig et al., *"Photo-Realistic Single Image Super-Resolution Using a GAN"*, CVPR 2017  
2. Wang et al., *"ESRGAN: Enhanced Super-Resolution GANs"*, ECCV 2018  
3. Dong et al., *"Image Super-Resolution Using Deep CNNs"*, TPAMI 2016  
