# ⚠️ Online Demo Coming Soon
The interactive Hugging Face Space for real-time inference is currently being deployed.  
Large model weights are still uploading and will be available shortly.

---

# Multimodal Face Forgery Detection

A robust multimodal framework for detecting manipulated facial media using adversarially trained vision–language models, prototype-based detection, and hybrid inference.

## Overview

This repository implements a forgery-aware multimodal detection system built on CLIP ViT-g-14 (LAION-2B).  
The framework jointly models facial imagery and textual descriptions to detect semantic and forensic inconsistencies under both clean and adversarial conditions.

The system includes:

- An adversarially trained multimodal contrastive backbone  
- Lightweight classifier heads for supervised detection  
- Prototype-based detectors for adversarial robustness  
- A hybrid fusion strategy combining classifier and prototype decisions  

The implementation supports research-grade evaluation and real-time deployment via Hugging Face Spaces.

---

## Architecture

### **Phase 1 — Multimodal Contrastive Backbone**
- CLIP ViT-g-14 image–text encoder  
- PGD-based adversarial training on images  
- Prompt ensemble for robust text encoding  
- Style conditioning for manipulation source and caption length  
- **ForensicToken**: Sobel-based high-pass forensic cues  
- **FiLMRefiner**: Cross-modal feature modulation  

This stage outputs an adversarially robust frozen backbone.

### **Phase 2 — Multimodal Classifier Heads**
Three supervised heads fine-tuned on the frozen backbone:

- **img_only** — image-only forgery detection  
- **i2t** — image → text semantic consistency  
- **t2i** — text → image semantic consistency  

Optional adversarial fine-tuning is supported.

---

## Prototype-Based Detection

Forgery prototypes are computed in:

- **Image embedding space**  
- **Pair embedding space**: `[img, txt, |img−txt|, img·txt]`  

These enable nearest-prototype detectors for both manipulation and mismatch detection.

---

## Hybrid Detection

- **Clean regime:** adaptive combination of classifier + prototype  
- **Adversarial regime:** prototype-only decision  
- Eliminates classifier flipping under attack  

---

## Repository Structure

### Training & Evaluation
- `train_contrastive.py` – backbone training  
- `train_classifier.py` – supervised head training  
- `compute_mm_prototypes.py` – prototype construction  
- `proto_classifier.py` – hybrid detector evaluation  
- `evaluate_backbone_proto.py` – backbone retrieval tests  
- `evaluate_classifier.py` – classifier evaluation  

### Core Modules
- `model_mm.py` – multimodal architecture  
- `dataset_mm.py` – dataset loader  
- `losses_mm.py` – training objectives  
- `utils_mm.py` – utilities  
- `logger.py` – logging  

### Deployment
- `inference_mm.py` – lightweight inference backend  
- `app.py` – Gradio interface for Hugging Face Space  

### Configuration
- `configs/config.yaml`  

---

## Training Pipeline

### 1. Backbone Training
```bash
python train_contrastive.py --config configs/backbone.yaml

