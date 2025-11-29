Multimodal Face Forgery Detection

A robust framework for detecting manipulated facial media using Vision-Language Models (VLM) with adversarial training.

Overview

This project leverages CLIP ViT-g-14 (LAION-2B) to build a forgery-aware multimodal backbone, enhanced with novel components for improved detection robustness. Features adversarially trained contrastive backbone resistant to image and text perturbations, lightweight classifier heads for multimodal inference, novelty modules including ForensicToken (edge-aware features) and FiLMRefiner (cross-modal conditioning), and hard negative mining with SBI augmentation for invariant feature learning.

Architecture

Phase 1: Contrastive Backbone Training
- Adversarial training with PGD attacks on images + text perturbations
- Prompt ensemble for robust text encoding
- Style adaptation for source/length conditioning
- Forensic analysis via Sobel-based high-pass features

Phase 2: Classifier Training  
- Frozen backbone with three specialized heads:
  - img_only: Image-only detection
  - i2t: Image-to-text pair analysis
  - t2i: Text-to-image pair analysis
- Optional adversarial training for enhanced robustness

Key Features

- Multimodal Robustness: Joint image-text understanding resistant to attacks
- Novel Components: ForensicToken (fixed Sobel filters + MLP for manipulation artifacts) and FiLMRefiner (cross-modal feature refinement)
- Data Augmentation: SBI blending + hard negative sampling
- Efficient Inference: Lightweight heads on frozen backbone

Usage

1. Backbone Training:
python train_contrastive.py --config configs/backbone.yaml

2. Classifier Training:
python train_classifier.py --config configs/classifier.yaml --init_from <backbone_ckpt>

Requirements

- PyTorch + CUDA
- OpenCLIP
- TIMM
- SentenceTransformers
- Weights & Biases (optional)

Results

Trained on FF++ dataset with cross-manipulation evaluation. Achieves state-of-the-art robustness against adversarial attacks while maintaining high detection accuracy across multiple forgery types.
