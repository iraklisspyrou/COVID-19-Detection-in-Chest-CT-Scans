# COVID-19 Detection in Chest CT Scans

**Custom CNN, ResNet18 & Vision Transformer (ViT-B/16)**  
Explainability via SmoothGradCAM++, MoRF & AOPC

---

## 📖 Overview

This project benchmarks three deep-learning architectures for binary COVID-19 classification on chest CT slices:

1. **CustomCNN**: A lightweight network (∼1.2 M parameters) trained from scratch  
2. **ResNet18**: An 18-layer residual network (11.2 M parameters) fine-tuned from ImageNet  
3. **ViT-B/16**: A Vision Transformer (∼86 M parameters) fine-tuned from ImageNet  

We demonstrate that transfer learning (especially ResNet18) delivers the best trade-off between accuracy and interpretability on a small, balanced dataset of 2,482 CT slices (1,252 COVID / 1,230 non-COVID).

---

## 🔍 Features

- **Classification Performance**:  
  - CustomCNN: 97.9 % accuracy (macro F1 0.9784)  
  - ResNet18: 98.9 % accuracy (macro F1 0.9892)  
  - ViT-B/16: 98.4 % accuracy (macro F1 0.9837)  

- **Explainability**  
  - **SmoothGradCAM++** heatmaps to visualize pixel‐level importance  
  - **MoRF** (Most Relevant First) protocol to perturb top-ranked pixels  
  - **AOPC** (Area Over the Perturbation Curve) to quantify explanation fidelity  

- **Reproducibility**  
  - Configurable training via `config.yaml`  
  - Scripts for data splitting, preprocessing, training, and evaluation  
  - Jupyter notebooks for interactive exploration  

---

---
## 📂 Project Structure

```
covid-ct-detection/
├── README.md                 # This file
├── config.yaml               # Hyperparameters (lr, epochs, splits, etc.)
├── split_data.py             # Script to shuffle & split dataset into train/val/test
├── transforms.py             # Definition of preprocessing and augmentation pipelines
├── data_loading.py           # PyTorch Dataset/DataLoader for CT slices
├── custom_cnn.py             # Implementation of the lightweight CNN from scratch
├── ResNet.py                 # Wrapper for fine-tuned ResNet18 (1-ch input → 2-class head)
├── vit_model.py              # Wrapper for fine-tuned ViT-B/16 (3-ch patch embedding)
├── main.py                   # Entry point: parses args, trains & evaluates a model
├── engine.py                 # Training/validation loops, checkpointing, early stopping
├── evaluate.py               # Computes test metrics (accuracy, F1, confusion matrix)
├── grad_cam.py               # SmoothGradCAM++ for CNN models
├── gradcam_vit.py            # SmoothGradCAM++ for ViT models
├── utils.py                  # Helper functions (seed setting, metric logging, plotting)
├── Report.pdf                # Analytic report of the project
└── Presentation.pptx         # Short presentation of the project
```
---
