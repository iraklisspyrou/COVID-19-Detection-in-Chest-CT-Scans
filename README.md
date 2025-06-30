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

