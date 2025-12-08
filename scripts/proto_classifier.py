#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid detector evaluation script.

Combines:
- Classifier head (img_only)
- Prototype-based detector (img-only prototypes)

Logic:
- On a small *clean* probe subset, compare clean AUC of classifier vs prototype.
- For CLEAN evaluation:
    - Use whichever has higher clean AUC on the probe (per-dataset decision).
- For ADVERSARIAL evaluation:
    - Always use the prototype stream (fixed rule).
- Report metrics for:
    - "cls"   : classifier-only
    - "proto" : prototype-only
    - "hyb"   : hybrid rule described above
"""

import argparse
from typing import Dict, Any, List

import timm

# ---------------------------------------------------------------------
# Disable timm pretrained downloads on HPC
# ---------------------------------------------------------------------
_real_create_model = timm.create_model


def _create_model_offline(*args, **kwargs):
    kwargs["pretrained"] = False
    return _real_create_model(*args, **kwargs)


timm.create_model = _create_model_offline

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_mm import load_config, seed_all
from dataset_mm import FFPPTripletDataset
from model_mm import MMModel
from train_classifier import (
    MMHeadsClassifier,
    collate_mm_classifier,
    pick_one_caption,
    pgd_attack_images_classifier,
)


# ---------------------------------------------------------------------
# Backbone construction
# ---------------------------------------------------------------------
def build_mm_backbone(cfg: Dict[str, Any], device: torch.device) -> MMModel:
    """
    Rebuild the multimodal backbone with the same configuration
    used during classifier training.
    """
    mm = MMModel(
        cfg["model"].get("img_dim", 512),
        cfg["model"].get("txt_dim", 384),
        cfg["model"].get("proj_dim", 256),
        cfg["model"].get("mini_lm_path", "data/mini_lm_embedder"),
        style_k=cfg["model"].get("style_k", 8),
        backbone=cfg["model"].get("backbone", "resnet50"),
        pretrained=bool(cfg["model"].get("pretrained", False)),
        model_type=cfg["model"].get("type", "clip"),
        clip_model_name=cfg["model"].get("clip_model_name", "ViT-g-14"),
        clip_pretrained_tag=cfg["model"].get("clip_pretrained_tag", None),
        clip_ckpt_path=cfg["model"].get("clip_ckpt_path", None),
        novelty_cfg=cfg.get("novelty", {}),
    ).to(device)
    return mm


# ---------------------------------------------------------------------
# Prototype scoring (image-only)
# ---------------------------------------------------------------------
def build_img_prototype_scorer(proto_path: str, device: torch.device):
    """
    Build a scoring function that maps image embeddings to a
    "fake margin" logit using image-only prototypes.

    Returns
    -------
    score_img_only : callable
        score_img_only(img_z) -> [B] tensor, higher = more fake.
    """
    proto_ckpt = torch.load(proto_path, map_location=device)

    if "prototypes_img" in proto_ckpt:
        # New format: explicit image prototypes
        protos_img = proto_ckpt["prototypes_img"].to(device)  # [2, D]
    else:
        # Backward-compatible: single "prototypes" key
        protos_img = proto_ckpt["prototypes"].to(device)

    p_img_real = F.normalize(protos_img[0], dim=-1)
    p_img_fake = F.normalize(protos_img[1], dim=-1)

    def score_img_only(img_z: torch.Tensor) -> torch.Tensor:
        """
        Prototype margin: (fake similarity - real similarity).

        Higher scores indicate stronger evidence for "fake".
        """
        z_img = F.normalize(img_z, dim=-1)
        s_real = (z_img * p_img_real).sum(dim=-1)
        s_fake = (z_img * p_img_fake).sum(dim=-1)
        return s_fake - s_real

    return score_img_only


# ---------------------------------------------------------------------
# Generic binary AUC helper
# ---------------------------------------------------------------------
def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Simple AUROC implementation for binary labels {0,1}, without sklearn.
    """
    scores, idx = torch.sort(scores, descending=True)
    labels = labels[idx]

    pos = labels.sum()
    neg = (1.0 - labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")

    tps = torch.cumsum(labels, dim=0)
    fps = torch.cumsum(1.0 - labels, dim=0)
    tpr = tps / pos
    fpr = fps / neg

    # prepend origin
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])

    return torch.trapz(tpr, fpr).item()


# ---------------------------------------------------------------------
# Decide best clean method (classifier vs prototype)
# ---------------------------------------------------------------------
def decide_best_clean_method(
    loader: DataLoader,
    clf_model: MMHeadsClassifier,
    mm: MMModel,
    score_proto_img,
    device: torch.device,
    templates: List[str],
    max_batches: int = 20,
):
    """
    Probe a subset of CLEAN data and compare classifier vs prototype.

    Decision rule:
      - If proto clean AUC >= classifier clean AUC:
            use_proto_for_clean = True
        else:
            use_proto_for_clean = False

    Returns
    -------
    use_proto_for_clean : bool
    auc_cls : float
    auc_proto : float
    """
    clf_model.eval()
    mm.eval()

    all_cls = []
    all_proto = []
    all_labels = []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        texts = pick_one_caption(batch["texts"])

        with torch.no_grad():
            # Classifier margins
            logits = clf_model(imgs, texts, templates, device, mode="img_only")
            m_cls = logits[:, 1] - logits[:, 0]

            # Prototype margins
            img_z = mm.encode_image(imgs)
            m_proto = score_proto_img(img_z)

        all_cls.append(m_cls.cpu())
        all_proto.append(m_proto.cpu())
        all_labels.append(labels.cpu())

    all_cls = torch.cat(all_cls)
    all_proto = torch.cat(all_proto)
    all_labels = torch.cat(all_labels).float()

    auc_cls = binary_auc(all_cls, all_labels)
    auc_proto = binary_auc(all_proto, all_labels)

    # Treat NaN as random (0.5)
    if auc_cls != auc_cls:
        auc_cls = 0.5
    if auc_proto != auc_proto:
        auc_proto = 0.5

    use_proto_for_clean = auc_proto >= auc_cls
    return use_proto_for_clean, auc_cls, auc_proto


# ---------------------------------------------------------------------
# Full eval: classifier / prototype / hybrid (clean + adv)
# Hybrid rule:
#   - Clean: proto or classifier based on decided AUC
#   - Adv:   always prototype
# ---------------------------------------------------------------------
def eval_clean_adv_with_decision(
    clf_model: MMHeadsClassifier,
    mm: MMModel,
    loader: DataLoader,
    device: torch.device,
    score_proto_img,
    templates: List[str],
    pgd_cfg: Dict[str, Any],
    use_proto_for_clean: bool,
    max_batches: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate:
      - 'cls'   : classifier-only (img_only head)
      - 'proto' : prototype-only (image prototypes)
      - 'hyb'   : hybrid rule
            CLEAN: best of {cls, proto} based on probe decision
            ADV:   prototype-only

    For each stream, computes:
      - clean_acc, adv_acc
      - clean_auc, adv_auc
      - fool_rate (clean-correct -> adv-incorrect)
      - flip_rate (prediction changed under attack)
      - total (number of samples)
    """
    clf_model.eval()
    mm.eval()

    stats = {
        "cls": {"clean_logits": [], "adv_logits": [], "labels": []},
        "proto": {"clean_logits": [], "adv_logits": [], "labels": []},
        "hyb": {"clean_logits": [], "adv_logits": [], "labels": []},
    }

    for bi, batch in enumerate(tqdm(loader, desc="Eval (clean+adv)")):
        if max_batches is not None and bi >= max_batches:
            break

        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()
        texts = pick_one_caption(batch["texts"])

        # ---- CLEAN ----
        with torch.no_grad():
            logits_clean = clf_model(imgs, texts, templates, device, mode="img_only")
            m_cls_clean = logits_clean[:, 1] - logits_clean[:, 0]

            img_z_clean = mm.encode_image(imgs)
            m_proto_clean = score_proto_img(img_z_clean)

            m_hyb_clean = m_proto_clean if use_proto_for_clean else m_cls_clean

        # ---- ADV (classifier-driven PGD on images) ----
        steps = int(pgd_cfg.get("steps", 0))
        if steps > 0:
            imgs_adv = pgd_attack_images_classifier(
                model=clf_model,
                imgs=imgs,
                labels=labels.long(),
                texts=texts,
                templates=templates,
                device=device,
                mode="img_only",
                eps=pgd_cfg.get("eps", 0.031),
                step_size=pgd_cfg.get("step_size", 0.0075),
                steps=steps,
                rand_start=True,
            )
        else:
            imgs_adv = imgs

        with torch.no_grad():
            logits_adv = clf_model(imgs_adv, texts, templates, device, mode="img_only")
            m_cls_adv = logits_adv[:, 1] - logits_adv[:, 0]

            img_z_adv = mm.encode_image(imgs_adv)
            m_proto_adv = score_proto_img(img_z_adv)

            # Adversarial: hybrid always uses prototype
            m_hyb_adv = m_proto_adv

        # Store per stream
        for key, mc, ma in [
            ("cls", m_cls_clean, m_cls_adv),
            ("proto", m_proto_clean, m_proto_adv),
            ("hyb", m_hyb_clean, m_hyb_adv),
        ]:
            stats[key]["clean_logits"].append(mc.cpu())
            stats[key]["adv_logits"].append(ma.cpu())
            stats[key]["labels"].append(labels.cpu())

    # -----------------------------------------------------------------
    # Aggregate and compute metrics
    # -----------------------------------------------------------------
    out: Dict[str, Dict[str, float]] = {}

    for key in ["cls", "proto", "hyb"]:
        clean_logits = torch.cat(stats[key]["clean_logits"])
        adv_logits = torch.cat(stats[key]["adv_logits"])
        labels_all = torch.cat(stats[key]["labels"])

        p_clean = torch.sigmoid(clean_logits)
        p_adv = torch.sigmoid(adv_logits)
        y = labels_all

        pred_clean = (p_clean >= 0.5).float()
        pred_adv = (p_adv >= 0.5).float()

        clean_acc = (pred_clean == y).float().mean().item()
        adv_acc = (pred_adv == y).float().mean().item()

        clean_correct = pred_clean == y
        adv_correct = pred_adv == y

        fools = clean_correct & (~adv_correct)
        fool_rate = fools.sum().item() / max(clean_correct.sum().item(), 1)

        flips = pred_clean != pred_adv
        flip_rate = flips.sum().item() / max(y.numel(), 1)

        auc_clean = binary_auc(p_clean, y)
        auc_adv = binary_auc(p_adv, y)

        out[key] = {
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "clean_auc": auc_clean,
            "adv_auc": auc_adv,
            "fool_rate": fool_rate,
            "flip_rate": flip_rate,
            "total": int(y.numel()),
        }

    return out


# ---------------------------------------------------------------------
# Argparse / main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (same as classifier training).")
    ap.add_argument("--clf_ckpt", required=True, help="Path to mm_heads_BEST_FULL.pt.")
    ap.add_argument(
        "--proto_path",
        default="outputs/forgery_prototypes_multi.pt",
        help="Path to prototype file (image-only or multi).",
    )
    ap.add_argument("--max_eval_batches", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.031, help="PGD epsilon (default ~8/255).")
    ap.add_argument(
        "--step_size",
        type=float,
        default=0.0075,
        help="PGD step size (default ~2/255).",
    )
    ap.add_argument("--steps", type=int, default=10, help="Number of PGD steps.")
    ap.add_argument(
        "--probe_batches",
        type=int,
        default=20,
        help="Number of clean batches used to decide fusion rule.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Backbone + classifier
    mm = build_mm_backbone(cfg, device)
    embed_dim = cfg.get("clf", {}).get("embed_dim", 1024)
    clf_model = MMHeadsClassifier(mm, embed_dim=embed_dim, share_pair_head=False).to(device)

    ckpt = torch.load(args.clf_ckpt, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = clf_model.load_state_dict(state_dict, strict=False)
    print(f"[INIT] Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}", flush=True)

    # Data
    data_cfg = cfg["data"]
    frames_root = data_cfg.get("frames_root", ".")
    project_root = data_cfg.get("project_root", ".")
    image_size = data_cfg.get("image_size", 224)

    val_ds = FFPPTripletDataset(
        jsonl_path=data_cfg["val_manifest"],
        frames_root=frames_root,
        image_size=image_size,
        augment=False,
        project_root=project_root,
        clip_norm=True,
        return_pair=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("clf", {}).get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("clf", {}).get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_mm_classifier,
    )

    templates = cfg["model"].get("clip_prompt_templates_eval", ["{caption}"])

    # Prototype scorer
    score_proto_img = build_img_prototype_scorer(args.proto_path, device)

    # 1) Decide clean fusion rule on a small clean probe
    print("\n[1] Deciding clean fusion rule on probe subset...")
    use_proto_for_clean, auc_cls, auc_proto = decide_best_clean_method(
        val_loader,
        clf_model,
        mm,
        score_proto_img,
        device,
        templates,
        max_batches=args.probe_batches,
    )
    decision_str = "Prototype" if use_proto_for_clean else "Classifier"
    print(f"    Classifier clean AUC: {auc_cls:.3f}")
    print(f"    Prototype  clean AUC: {auc_proto:.3f}")
    print(f"    Decision for CLEAN: {decision_str}")
    print("    Decision for ADV:   Prototype (fixed)\n")

    # 2) Full clean+adv evaluation under the fixed rule
    pgd_cfg = {"eps": args.eps, "step_size": args.step_size, "steps": args.steps}
    print("[2] Running full clean+adv evaluation with fixed rule...")
    metrics = eval_clean_adv_with_decision(
        clf_model,
        mm,
        val_loader,
        device,
        score_proto_img,
        templates,
        pgd_cfg=pgd_cfg,
        use_proto_for_clean=use_proto_for_clean,
        max_batches=args.max_eval_batches,
    )

    print("\n" + "=" * 80)
    print("ADAPTIVE FUSION RESULTS")
    print("  - Clean: per-dataset best of {classifier, prototype}")
    print("  - Adv:   prototype-only")
    print("=" * 80)

    for name, m in metrics.items():
        print(
            f"[{name}] "
            f"clean_acc={m['clean_acc']:.3f} adv_acc={m['adv_acc']:.3f} "
            f"clean_auc={m['clean_auc']:.3f} adv_auc={m['adv_auc']:.3f} "
            f"fool_rate={m['fool_rate']:.3f} flip_rate={m['flip_rate']:.3f} "
            f"total={m['total']}",
            flush=True,
        )

    print("\n" + "=" * 80)
    print("SUMMARY (intuition):")
    print("  - Hybrid clean performance ≈ max(clean AUC of classifier, prototype)")
    print("  - Hybrid adversarial performance ≈ prototype stream")
    print("  - Classifier stream remains visible for comparison in results")
    print("=" * 80)


if __name__ == "__main__":
    main()

