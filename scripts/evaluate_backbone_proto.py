# scripts/eval_adv_only.py
# -*- coding: utf-8 -*-
"""
Standalone evaluation script for a saved FULL multimodal checkpoint.

Provides:
- Clean retrieval evaluation for the multimodal backbone (eval_clean_all)
- Adversarial retrieval evaluation for the backbone (run_adv_eval / eval_adv_all_simple)
- Prototype-based detector evaluation (clean + adversarial) using precomputed prototypes

This script is intended to be used offline for reporting and analysis.
"""

import os
import sys
import gc
import argparse
import warnings
from itertools import islice
from typing import Dict, Any, Callable

warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ---------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from train_contrastive import (
    load_config,
    seed_all,
    FFPPTripletDataset,
    MMModel,
    collate_mm,
    encode_with_prompt_ensemble,
    eval_adv_all,
    build_text_batch_for_it_balanced,
    encode_variants_with_prompts,
    make_two_text_views,
    cosine_sim,
    r_at_1_from_sim,
    r_at_5_from_sim,
    median_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    auc_cmc,
    pairwise_auroc,
    pgd_attack_images,
)

# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove leading 'module.' from state_dict keys if present."""
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_full_checkpoint(
    model: nn.Module,
    full_ckpt_path: str,
    device: torch.device,
):
    """
    Load a FULL checkpoint (including backbone + projector + novelty modules).

    Returns
    -------
    mm : nn.Module
        The model with weights loaded and frozen in eval mode.
    ema_shadow : dict | None
        EMA shadow weights, if present in the checkpoint.
    epoch_abs_from_ckpt : int | None
        The absolute epoch index stored in the checkpoint, if any.
    """
    print(f"[INIT] Loading FULL ckpt for eval: {full_ckpt_path}", flush=True)
    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    sd_raw = ckpt.get("model", ckpt)
    sd = _strip_module_prefix(sd_raw)

    target = model.module if isinstance(model, nn.DataParallel) else model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(
        f"[INIT] load_state: missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    ema_shadow = ckpt.get("ema_shadow", None)
    if ema_shadow is not None:
        ema_shadow = {k: v.to(device) for k, v in ema_shadow.items()}

    epoch_abs_from_ckpt = ckpt.get("epoch", None)

    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    mm = target
    return mm, ema_shadow, epoch_abs_from_ckpt


def restore_ema_weights(mm: nn.Module, ema_shadow: Dict[str, torch.Tensor] | None) -> None:
    """Apply EMA shadow weights to a model if available."""
    if ema_shadow is None:
        print("[EMA] No EMA shadow in checkpoint, evaluating raw weights.", flush=True)
        return

    print("[EMA] Applying EMA shadow weights for eval...", flush=True)
    with torch.no_grad():
        for n, p in mm.named_parameters():
            if n in ema_shadow:
                p.copy_(ema_shadow[n])


# ---------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------


def build_val_loader(cfg: Dict[str, Any]) -> DataLoader:
    """Build the validation/test loader used for evaluation."""
    model_type = cfg["model"].get("type", "clip")
    clip_norm = model_type == "clip"

    val_ds = FFPPTripletDataset(
        cfg["data"]["val_manifest"],
        cfg["data"]["frames_root"],
        image_size=cfg["data"].get("image_size", 224),
        augment=False,
        clip_norm=clip_norm,
        return_pair=True,
    )

    numw = min(int(cfg["train"].get("num_workers", 4)), 8)
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))

    def _worker_init_fn(worker_id: int) -> None:
        torch.manual_seed(cfg.get("seed", 42) + worker_id)

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=numw,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_mm,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    return val_loader


# ---------------------------------------------------------------------
# Backbone: adversarial eval (simple helper, optional)
# ---------------------------------------------------------------------


@torch.no_grad()
def eval_adv_all_simple(
    mm: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    tpl_eval=None,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """
    Simple adversarial retrieval evaluation that mirrors the clean evaluation,
    but applies PGD to the anchor images only.

    This mirrors the clean metrics (I2T, T2I, I2I, T2T, plus robust),
    returning keys with *_adv suffix. It is not used by default in main()
    but can be imported and called from other scripts/notebooks.
    """
    if tpl_eval is None:
        tpl_eval = ["{caption}"]

    model = mm.module if isinstance(mm, nn.DataParallel) else mm
    model.eval()

    # Same accumulators as clean, but with _adv suffix
    R1_i2t = R5_i2t = MR_i2t = MRR_i2t = NDCG5_i2t = AUCMC_i2t = AUROC_i2t = 0.0
    R1_t2i = R5_t2i = MR_t2i = MRR_t2i = NDCG5_t2i = AUCMC_t2i = AUROC_t2i = 0.0
    R1_i2i = R5_i2i = MR_i2i = MRR_i2i = NDCG5_i2i = AUCMC_i2i = AUROC_i2i = 0.0
    R1_t2t = R5_t2t = MR_t2t = MRR_t2t = NDCG5_t2t = AUCMC_t2t = AUROC_t2t = 0.0

    vc = 0
    robust_batches = 0
    R1_rob = R5_rob = MR_rob = AUROC_rob = 0.0

    eval_adv_cfg = cfg.get("eval_adv", {}).get("pgd", {})
    steps = int(eval_adv_cfg.get("steps", 0))

    base_iter = loader
    if max_batches is not None and max_batches > 0:
        base_iter = islice(loader, max_batches)

    it = tqdm(
        base_iter,
        desc="AdvEvalSimple",
        total=max_batches
        if max_batches is not None
        else (len(loader) if hasattr(loader, "__len__") else None),
        ncols=60,
        dynamic_ncols=False,
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        mininterval=1.0,
        smoothing=0,
    )

    for batch in it:
        imgs = batch["image"].to(device, non_blocking=True)

        imgs2 = batch.get("image2", None)
        if isinstance(imgs2, torch.Tensor):
            imgs2 = imgs2.to(device, non_blocking=True)
        else:
            imgs2 = None

        blend = batch.get("blend_image", None)
        if isinstance(blend, torch.Tensor):
            blend = blend.to(device, non_blocking=True)
        else:
            blend = None

        texts_list = batch["texts"]
        picked = build_text_batch_for_it_balanced(texts_list, step=0)

        # PGD on images
        if steps > 0:
            imgs_adv = pgd_attack_images(
                model,
                imgs,
                picked,
                cfg,
                device,
                tpl_eval,
                target_emb=None,
                objective="i2t",  # same objective as main eval
                eval_mode=True,
            )
        else:
            imgs_adv = imgs

        # Encode adv image / clean text
        img_z_raw = model.encode_image(imgs_adv)
        txt_z_raw = encode_with_prompt_ensemble(model, picked, tpl_eval, device)

        refiner = getattr(model, "refine_embeddings", None)
        if callable(refiner):
            img_z, txt_z = refiner(img_z_raw, txt_z_raw, imgs_adv)
        else:
            img_z, txt_z = img_z_raw, txt_z_raw

        # T2T metrics
        variant_embs = encode_variants_with_prompts(model, texts_list, tpl_eval, device)
        txtA, txtB = make_two_text_views(variant_embs)
        if txtA is None or txtB is None or txtA.size(0) != img_z.size(0):
            txtA = encode_with_prompt_ensemble(model, picked, tpl_eval, device)
            txtB = encode_with_prompt_ensemble(model, picked, tpl_eval, device)

        sim_tt = cosine_sim(txtA, txtB)
        R1_t2t += r_at_1_from_sim(sim_tt)
        R5_t2t += r_at_5_from_sim(sim_tt)
        MR_t2t += median_rank(sim_tt)
        MRR_t2t += mean_reciprocal_rank(sim_tt)
        NDCG5_t2t += ndcg_at_k(sim_tt, 5)
        AUCMC_t2t += auc_cmc(sim_tt)
        AUROC_t2t += pairwise_auroc(sim_tt)

        # I2I metrics (if positive image pair)
        img_pos = None
        blend_z = None
        imgs2_z = None

        if isinstance(blend, torch.Tensor):
            blend_z = model.encode_image(blend)
            img_pos = blend_z
        elif isinstance(imgs2, torch.Tensor):
            imgs2_z = model.encode_image(imgs2)
            img_pos = imgs2_z

        # Robust "max" retrieval
        if blend_z is not None:
            sim_img = cosine_sim(img_z, txt_z)
            sim_bl = cosine_sim(blend_z, txt_z)
            sim_max = torch.maximum(sim_img, sim_bl)

            R1_rob += r_at_1_from_sim(sim_max)
            R5_rob += r_at_5_from_sim(sim_max)
            MR_rob += median_rank(sim_max)
            AUROC_rob += pairwise_auroc(sim_max)
            robust_batches += 1

        if img_pos is not None:
            sim_ii = cosine_sim(img_z, img_pos)
            R1_i2i += r_at_1_from_sim(sim_ii)
            R5_i2i += r_at_5_from_sim(sim_ii)
            MR_i2i += median_rank(sim_ii)
            MRR_i2i += mean_reciprocal_rank(sim_ii)
            NDCG5_i2i += ndcg_at_k(sim_ii, 5)
            AUCMC_i2i += auc_cmc(sim_ii)
            AUROC_i2i += pairwise_auroc(sim_ii)

        # I2T / T2I metrics
        sim = cosine_sim(img_z, txt_z)
        R1_i2t += r_at_1_from_sim(sim)
        R5_i2t += r_at_5_from_sim(sim)
        MR_i2t += median_rank(sim)
        MRR_i2t += mean_reciprocal_rank(sim)
        NDCG5_i2t += ndcg_at_k(sim, 5)
        AUCMC_i2t += auc_cmc(sim)
        AUROC_i2t += pairwise_auroc(sim)

        simT = sim.T
        R1_t2i += r_at_1_from_sim(simT)
        R5_t2i += r_at_5_from_sim(simT)
        MR_t2i += median_rank(simT)
        MRR_t2i += mean_reciprocal_rank(simT)
        NDCG5_t2i += ndcg_at_k(simT, 5)
        AUCMC_t2i += auc_cmc(simT)
        AUROC_t2i += pairwise_auroc(simT)

        vc += 1

    safe_div = lambda x: x / max(vc, 1)
    safe_div_rob = lambda x: x / max(robust_batches, 1) if robust_batches > 0 else 0.0

    adv_metrics = {
        # I2T
        "R@1_i2t_adv": safe_div(R1_i2t),
        "R@5_i2t_adv": safe_div(R5_i2t),
        "MR_i2t_adv": safe_div(MR_i2t),
        "MRR_i2t_adv": safe_div(MRR_i2t),
        "NDCG@5_i2t_adv": safe_div(NDCG5_i2t),
        "AUCMC_i2t_adv": safe_div(AUCMC_i2t),
        "AUROC_i2t_adv": safe_div(AUROC_i2t),
        # T2I
        "R@1_t2i_adv": safe_div(R1_t2i),
        "R@5_t2i_adv": safe_div(R5_t2i),
        "MR_t2i_adv": safe_div(MR_t2i),
        "MRR_t2i_adv": safe_div(MRR_t2i),
        "NDCG@5_t2i_adv": safe_div(NDCG5_t2i),
        "AUCMC_t2i_adv": safe_div(AUCMC_t2i),
        "AUROC_t2i_adv": safe_div(AUROC_t2i),
        # I2I
        "R@1_i2i_adv": safe_div(R1_i2i),
        "R@5_i2i_adv": safe_div(R5_i2i),
        "MR_i2i_adv": safe_div(MR_i2i),
        "MRR_i2i_adv": safe_div(MRR_i2i),
        "NDCG@5_i2i_adv": safe_div(NDCG5_i2i),
        "AUCMC_i2i_adv": safe_div(AUCMC_i2i),
        "AUROC_i2i_adv": safe_div(AUROC_i2i),
        # T2T
        "R@1_t2t_adv": safe_div(R1_t2t),
        "R@5_t2t_adv": safe_div(R5_t2t),
        "MR_t2t_adv": safe_div(MR_t2t),
        "MRR_t2t_adv": safe_div(MRR_t2t),
        "NDCG@5_t2t_adv": safe_div(NDCG5_t2t),
        "AUCMC_t2t_adv": safe_div(AUCMC_t2t),
        "AUROC_t2t_adv": safe_div(AUROC_t2t),
        # Robust (optional)
        "R@1_rob_adv": safe_div_rob(R1_rob),
        "R@5_rob_adv": safe_div_rob(R5_rob),
        "MR_rob_adv": safe_div_rob(MR_rob),
        "AUROC_rob_adv": safe_div_rob(AUROC_rob),
    }

    return adv_metrics


def run_adv_eval(
    cfg: Dict[str, Any],
    mm: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
    epoch_abs: int,
    epoch_rel: int,
) -> Dict[str, float]:
    """
    Wrapper that calls the original eval_adv_all (from training script)
    with a limited number of validation batches.
    """
    tpl_eval = cfg["model"].get("clip_prompt_templates_eval", ["{caption}"])
    text_cfg_eval = cfg.get("eval_adv", {}).get("text", {})
    max_adv_val_batches = int(cfg.get("eval_adv", {}).get("max_val_batches", 64))

    print(f"[EvalAdv] Starting adversarial evaluation at epoch {epoch_abs} ...", flush=True)

    adv_cfg = cfg.get("eval_adv", {}).get("pgd", {})
    limited_val_iter = islice(val_loader, max_adv_val_batches)

    adv_all_metrics = eval_adv_all(
        mm,
        limited_val_iter,
        device,
        tpl_eval,
        adv_cfg,
        text_cfg_eval,
        cfg,
        total=max_adv_val_batches,
        desc=f"Adv{epoch_abs}",
    )

    print(f"[EvalAdv] Finished epoch {epoch_abs}", flush=True)

    r_i2t = adv_all_metrics.get("R@1_i2t_adv", float("nan"))
    r_t2i = adv_all_metrics.get("R@1_t2i_adv", float("nan"))
    r_i2i = adv_all_metrics.get("R@1_i2i_adv", float("nan"))
    r_t2t = adv_all_metrics.get("R@1_t2t_adv", float("nan"))

    print(
        "[ValAdvSummary] "
        f"epoch_abs={epoch_abs} rel={epoch_rel} | "
        "ADV R@1 i2t/t2i/i2i/t2t="
        f"{r_i2t:.3f}/{r_t2i:.3f}/{r_i2i:.3f}/{r_t2t:.3f}",
        flush=True,
    )
    print(
        "[ValAdvFullMetrics]",
        {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v for k, v in adv_all_metrics.items()},
        flush=True,
    )

    return adv_all_metrics


# ---------------------------------------------------------------------
# Backbone: clean retrieval evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def eval_clean_all(
    mm: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    tpl_eval=None,
) -> Dict[str, float]:
    """
    Clean retrieval evaluation (no PGD, no text noise).

    Returns metrics with *_clean suffix, e.g.:
    - R@1_i2t_clean, R@1_t2i_clean, R@1_i2i_clean, R@1_t2t_clean
    - plus robust retrieval metrics using the blended image when available.
    """
    if tpl_eval is None:
        tpl_eval = ["{caption}"]

    model = mm.module if isinstance(mm, nn.DataParallel) else mm
    model.eval()

    # Accumulators (match training validation)
    R1_i2t = R5_i2t = MR_i2t = MRR_i2t = NDCG5_i2t = AUCMC_i2t = AUROC_i2t = 0.0
    R1_t2i = R5_t2i = MR_t2i = MRR_t2i = NDCG5_t2i = AUCMC_t2i = AUROC_t2i = 0.0
    R1_i2i = R5_i2i = MR_i2i = MRR_i2i = NDCG5_i2i = AUCMC_i2i = AUROC_i2i = 0.0
    R1_t2t = R5_t2t = MR_t2t = MRR_t2t = NDCG5_t2t = AUCMC_t2t = AUROC_t2t = 0.0

    R1_rob = R5_rob = MR_rob = AUROC_rob = 0.0
    robust_batches = 0

    vc = 0

    it = tqdm(
        loader,
        desc="CleanEval",
        total=len(loader) if hasattr(loader, "__len__") else None,
        ncols=60,
        dynamic_ncols=False,
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        mininterval=1.0,
        smoothing=0,
    )

    for batch in it:
        imgs = batch["image"].to(device, non_blocking=True)

        imgs2 = batch.get("image2", None)
        if isinstance(imgs2, torch.Tensor):
            imgs2 = imgs2.to(device, non_blocking=True)
        else:
            imgs2 = None

        blend = batch.get("blend_image", None)
        if isinstance(blend, torch.Tensor):
            blend = blend.to(device, non_blocking=True)
        else:
            blend = None

        texts_list = batch["texts"]

        picked = build_text_batch_for_it_balanced(texts_list, step=0)

        # Encode image / text
        img_z_raw = model.encode_image(imgs)
        txt_z_raw = encode_with_prompt_ensemble(model, picked, tpl_eval, device)

        refiner = getattr(model, "refine_embeddings", None)
        if callable(refiner):
            img_z, txt_z = refiner(img_z_raw, txt_z_raw, imgs)
        else:
            img_z, txt_z = img_z_raw, txt_z_raw

        # T2T metrics
        variant_embs = encode_variants_with_prompts(model, texts_list, tpl_eval, device)
        txtA, txtB = make_two_text_views(variant_embs)
        if txtA is None or txtB is None or txtA.size(0) != img_z.size(0):
            txtA = encode_with_prompt_ensemble(model, picked, tpl_eval, device)
            txtB = encode_with_prompt_ensemble(model, picked, tpl_eval, device)

        sim_tt = cosine_sim(txtA, txtB)
        R1_t2t += r_at_1_from_sim(sim_tt)
        R5_t2t += r_at_5_from_sim(sim_tt)
        MR_t2t += median_rank(sim_tt)
        MRR_t2t += mean_reciprocal_rank(sim_tt)
        NDCG5_t2t += ndcg_at_k(sim_tt, 5)
        AUCMC_t2t += auc_cmc(sim_tt)
        AUROC_t2t += pairwise_auroc(sim_tt)

        # I2I metrics (if positive pair)
        img_pos = None
        blend_z = None

        if isinstance(blend, torch.Tensor):
            blend_z = model.encode_image(blend)
            img_pos = blend_z
        elif isinstance(imgs2, torch.Tensor):
            img_pos = model.encode_image(imgs2)

        # Robust "max" retrieval (blend vs clean)
        if blend_z is not None:
            sim_img = cosine_sim(img_z, txt_z)
            sim_bl = cosine_sim(blend_z, txt_z)
            sim_max = torch.maximum(sim_img, sim_bl)

            R1_rob += r_at_1_from_sim(sim_max)
            R5_rob += r_at_5_from_sim(sim_max)
            MR_rob += median_rank(sim_max)
            AUROC_rob += pairwise_auroc(sim_max)
            robust_batches += 1

        if img_pos is not None:
            sim_ii = cosine_sim(img_z, img_pos)
            R1_i2i += r_at_1_from_sim(sim_ii)
            R5_i2i += r_at_5_from_sim(sim_ii)
            MR_i2i += median_rank(sim_ii)
            MRR_i2i += mean_reciprocal_rank(sim_ii)
            NDCG5_i2i += ndcg_at_k(sim_ii, 5)
            AUCMC_i2i += auc_cmc(sim_ii)
            AUROC_i2i += pairwise_auroc(sim_ii)

        # I2T / T2I metrics
        sim = cosine_sim(img_z, txt_z)
        R1_i2t += r_at_1_from_sim(sim)
        R5_i2t += r_at_5_from_sim(sim)
        MR_i2t += median_rank(sim)
        MRR_i2t += mean_reciprocal_rank(sim)
        NDCG5_i2t += ndcg_at_k(sim, 5)
        AUCMC_i2t += auc_cmc(sim)
        AUROC_i2t += pairwise_auroc(sim)

        simT = sim.T
        R1_t2i += r_at_1_from_sim(simT)
        R5_t2i += r_at_5_from_sim(simT)
        MR_t2i += median_rank(simT)
        MRR_t2i += mean_reciprocal_rank(simT)
        NDCG5_t2i += ndcg_at_k(simT, 5)
        AUCMC_t2i += auc_cmc(simT)
        AUROC_t2i += pairwise_auroc(simT)

        vc += 1

    safe_div = lambda x: x / max(vc, 1)
    safe_div_rob = lambda x: x / max(robust_batches, 1) if robust_batches > 0 else 0.0

    clean_metrics = {
        # I2T
        "R@1_i2t_clean": safe_div(R1_i2t),
        "R@5_i2t_clean": safe_div(R5_i2t),
        "MR_i2t_clean": safe_div(MR_i2t),
        "MRR_i2t_clean": safe_div(MRR_i2t),
        "NDCG@5_i2t_clean": safe_div(NDCG5_i2t),
        "AUCMC_i2t_clean": safe_div(AUCMC_i2t),
        "AUROC_i2t_clean": safe_div(AUROC_i2t),
        # T2I
        "R@1_t2i_clean": safe_div(R1_t2i),
        "R@5_t2i_clean": safe_div(R5_t2i),
        "MR_t2i_clean": safe_div(MR_t2i),
        "MRR_t2i_clean": safe_div(MRR_t2i),
        "NDCG@5_t2i_clean": safe_div(NDCG5_t2i),
        "AUCMC_t2i_clean": safe_div(AUCMC_t2i),
        "AUROC_t2i_clean": safe_div(AUROC_t2i),
        # I2I
        "R@1_i2i_clean": safe_div(R1_i2i),
        "R@5_i2i_clean": safe_div(R5_i2i),
        "MR_i2i_clean": safe_div(MR_i2i),
        "MRR_i2i_clean": safe_div(MRR_i2i),
        "NDCG@5_i2i_clean": safe_div(NDCG5_i2i),
        "AUCMC_i2i_clean": safe_div(AUCMC_i2i),
        "AUROC_i2i_clean": safe_div(AUROC_i2i),
        # T2T
        "R@1_t2t_clean": safe_div(R1_t2t),
        "R@5_t2t_clean": safe_div(R5_t2t),
        "MR_t2t_clean": safe_div(MR_t2t),
        "MRR_t2t_clean": safe_div(MRR_t2t),
        "NDCG@5_t2t_clean": safe_div(NDCG5_t2t),
        "AUCMC_t2t_clean": safe_div(AUCMC_t2t),
        "AUROC_t2t_clean": safe_div(AUROC_t2t),
        # Robust
        "R@1_rob_clean": safe_div_rob(R1_rob),
        "R@5_rob_clean": safe_div_rob(R5_rob),
        "MR_rob_clean": safe_div_rob(MR_rob),
        "AUROC_rob_clean": safe_div_rob(AUROC_rob),
    }

    return clean_metrics


# ---------------------------------------------------------------------
# Prototype detector: clean + adversarial evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def eval_detector_clean_adv_multi(
    mm: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    tpl_eval,
    score_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    max_batches: int | None = None,
    fixed_theta: float | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple prototype-based detectors (e.g. img_only, pair) in one pass:

      - clean images
      - PGD-adversarial images (same PGD config as retrieval eval)

    Parameters
    ----------
    mm : nn.Module
        Multimodal backbone (with optional refiner).
    loader : DataLoader
        Validation/test loader.
    device : torch.device
    cfg : dict
        Config dictionary (with eval_adv.pgd, etc.).
    tpl_eval : list[str]
        Prompt templates for CLIP text encoding.
    score_fns : dict[str, callable]
        Mapping from mode name -> score function(img_z, txt_z) -> [B] logit (higher = more fake).
    max_batches : int | None
        Optional cap on number of batches to use.
    fixed_theta : float | None
        Optional fixed threshold. If None, a simple grid search per mode is used.

    Returns
    -------
    out : dict
        Per-mode metrics:
          {
            mode: {
              det_theta,
              det_acc_clean,
              det_auc_clean,
              det_acc_adv,
              det_auc_adv,
              det_fool_rate,
              det_flip_rate,
              det_total,
            },
            ...
          }
    """
    model = mm.module if isinstance(mm, nn.DataParallel) else mm
    model.eval()

    eval_adv_cfg = cfg.get("eval_adv", {}).get("pgd", {})
    steps = int(eval_adv_cfg.get("steps", 0))

    if max_batches is not None and max_batches > 0:
        base_iter = islice(loader, max_batches)
        total = max_batches
    else:
        base_iter = loader
        total = len(loader) if hasattr(loader, "__len__") else None

    it = tqdm(
        base_iter,
        desc="DetEval(clean+adv)",
        total=total,
        ncols=60,
        dynamic_ncols=False,
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        mininterval=1.0,
        smoothing=0,
    )

    all_scores_clean = {name: [] for name in score_fns}
    all_scores_adv = {name: [] for name in score_fns}
    all_labels = []

    for batch in it:
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)  # 0=real, 1=fake
        texts_list = batch["texts"]

        picked = build_text_batch_for_it_balanced(texts_list, step=0)
        txt_z_raw = encode_with_prompt_ensemble(model, picked, tpl_eval, device)

        # Clean embeddings
        img_z_raw_clean = model.encode_image(imgs)
        refiner = getattr(model, "refine_embeddings", None)
        if callable(refiner):
            img_z_clean, txt_z_clean = refiner(img_z_raw_clean, txt_z_raw, imgs)
        else:
            img_z_clean, txt_z_clean = img_z_raw_clean, txt_z_raw

        for name, fn in score_fns.items():
            s_clean = fn(img_z_clean, txt_z_clean)  # [B]
            all_scores_clean[name].append(s_clean.detach().cpu())

        # Adversarial embeddings
        if steps > 0:
            imgs_adv = pgd_attack_images(
                model,
                imgs,
                picked,
                cfg,
                device,
                tpl_eval,
                target_emb=None,
                objective="i2t",
                eval_mode=True,
            )
        else:
            imgs_adv = imgs

        img_z_raw_adv = model.encode_image(imgs_adv)
        if callable(refiner):
            img_z_adv, txt_z_adv = refiner(img_z_raw_adv, txt_z_raw, imgs_adv)
        else:
            img_z_adv, txt_z_adv = img_z_raw_adv, txt_z_raw

        for name, fn in score_fns.items():
            s_adv = fn(img_z_adv, txt_z_adv)  # [B]
            all_scores_adv[name].append(s_adv.detach().cpu())

        all_labels.append(labels.detach().cpu())

    # Stack everything
    all_labels_tensor = torch.cat(all_labels).float()  # [N]

    # Metric helpers ------------------------------------------------

    def compute_metrics_for_threshold(
        scores_clean: torch.Tensor,
        scores_adv: torch.Tensor,
        labels: torch.Tensor,
        theta: float,
    ):
        preds_clean = (scores_clean >= theta).float()
        preds_adv = (scores_adv >= theta).float()

        acc_clean = (preds_clean == labels).float().mean().item()
        acc_adv = (preds_adv == labels).float().mean().item()

        clean_correct = preds_clean == labels
        adv_correct = preds_adv == labels

        fools_mask = clean_correct & (~adv_correct)
        flips_mask = preds_clean != preds_adv

        num_correct_clean = clean_correct.sum().item()
        num_fools = fools_mask.sum().item()
        num_flips = flips_mask.sum().item()
        total = int(labels.numel())

        fool_rate = num_fools / max(num_correct_clean, 1)
        flip_rate = num_flips / max(total, 1)

        return acc_clean, acc_adv, fool_rate, flip_rate

    def binary_auc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
        # Manual AUROC implementation (no sklearn dependency)
        sorted_scores, indices = torch.sort(scores, descending=True)
        sorted_labels = labels[indices]

        pos = sorted_labels.sum()
        neg = (1.0 - sorted_labels).sum()
        if pos == 0 or neg == 0:
            return float("nan")

        tps = torch.cumsum(sorted_labels, dim=0)
        fps = torch.cumsum(1.0 - sorted_labels, dim=0)
        tpr = tps / pos
        fpr = fps / neg

        tpr = torch.cat([torch.tensor([0.0]), tpr])
        fpr = torch.cat([torch.tensor([0.0]), fpr])

        return torch.trapz(tpr, fpr).item()

    target_clean = cfg.get("detector", {}).get("target_clean_acc", 0.85)

    # Per-mode threshold search + metrics ---------------------------
    out: Dict[str, Dict[str, float]] = {}
    for name in score_fns.keys():
        scores_clean = torch.cat(all_scores_clean[name])
        scores_adv = torch.cat(all_scores_adv[name])

        if fixed_theta is None:
            min_s = float(scores_clean.min())
            max_s = float(scores_clean.max())
            thetas = torch.linspace(min_s, max_s, steps=100)

            best_obj = -1e9
            best_theta = 0.0
            best_metrics = (0.0, 0.0, 1.0, 1.0)

            for theta in thetas:
                acc_clean, acc_adv, fool_rate, flip_rate = compute_metrics_for_threshold(
                    scores_clean, scores_adv, all_labels_tensor, float(theta)
                )

                penalty = 0.0 if acc_clean >= target_clean else -10.0
                obj = acc_adv + penalty

                if obj > best_obj:
                    best_obj = obj
                    best_theta = float(theta)
                    best_metrics = (acc_clean, acc_adv, fool_rate, flip_rate)

            det_theta = best_theta
            det_acc_clean, det_acc_adv, det_fool_rate, det_flip_rate = best_metrics
        else:
            det_theta = float(fixed_theta)
            det_acc_clean, det_acc_adv, det_fool_rate, det_flip_rate = compute_metrics_for_threshold(
                scores_clean, scores_adv, all_labels_tensor, det_theta
            )

        det_auc_clean = binary_auc_from_scores(scores_clean, all_labels_tensor)
        det_auc_adv = binary_auc_from_scores(scores_adv, all_labels_tensor)

        metrics = {
            "det_theta": det_theta,
            "det_acc_clean": det_acc_clean,
            "det_auc_clean": det_auc_clean,
            "det_acc_adv": det_acc_adv,
            "det_auc_adv": det_auc_adv,
            "det_fool_rate": det_fool_rate,
            "det_flip_rate": det_flip_rate,
            "det_total": int(all_labels_tensor.numel()),
        }
        print(f"[DetEval][{name}] {metrics}", flush=True)
        out[name] = metrics

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (same one used in training).",
    )
    ap.add_argument(
        "--full_ckpt",
        required=True,
        help="Path to the *_FULL.pt checkpoint to evaluate.",
    )
    ap.add_argument(
        "--epoch_abs",
        type=int,
        default=-1,
        help="Optional epoch_abs to print in logs.",
    )
    ap.add_argument(
        "--epoch_rel",
        type=int,
        default=-1,
        help="Optional epoch_rel to print in logs.",
    )
    ap.add_argument(
        "--proto_path",
        type=str,
        default="outputs/forgery_prototypes.pt",
        help="Path to saved prototype file.",
    )
    ap.add_argument(
        "--fixed_theta",
        type=float,
        default=None,
        help="Optional fixed threshold; if omitted, grid search is used per mode.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg.get("seed", 42))

    # Optionally merge eval_adv settings from checkpoint
    ckpt_raw = torch.load(args.full_ckpt, map_location="cpu")
    ckpt_cfg = ckpt_raw.get("cfg", None)

    if ckpt_cfg is not None:
        if "eval_adv" in ckpt_cfg:
            cfg.setdefault("eval_adv", {})
            for k, v in ckpt_cfg["eval_adv"].items():
                if k == "max_val_batches":
                    continue
                cfg["eval_adv"][k] = v

        if "model" in ckpt_cfg and "clip_prompt_templates_eval" in ckpt_cfg["model"]:
            cfg.setdefault("model", {})
            cfg["model"]["clip_prompt_templates_eval"] = ckpt_cfg["model"]["clip_prompt_templates_eval"]
    else:
        print("[WARN] No 'cfg' found in checkpoint; using YAML eval_adv as-is.", flush=True)

    cfg.setdefault("train", {})
    cfg["train"]["batch_size"] = int(cfg["train"].get("batch_size", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional W&B init (for logging metrics)
    if WANDB_AVAILABLE:
        exp_cfg = cfg.get("exp", {})
        use_wandb = exp_cfg.get("wandb", True)

        if use_wandb:
            run_name = f"eval_adv_{os.path.basename(args.full_ckpt).replace('.pt', '')}"
            project = exp_cfg.get("project", exp_cfg.get("name", "mm_clip_eval"))
            wandb.init(
                project=project,
                name=run_name,
                config=cfg,
                mode=os.environ.get("WANDB_MODE", "offline"),
            )

    # Build backbone
    model_type = cfg["model"].get("type", "clip")
    mm_model = MMModel(
        cfg["model"].get("img_dim", 512),
        cfg["model"].get("txt_dim", 384),
        cfg["model"].get("proj_dim", 256),
        cfg["model"].get("mini_lm_path", "data/mini_lm_embedder"),
        style_k=cfg["model"].get("style_k", 8),
        backbone=cfg["model"].get("backbone", "resnet50"),
        pretrained=bool(cfg["model"].get("pretrained", False)),
        model_type=model_type,
        clip_model_name=cfg["model"].get("clip_model_name", "ViT-g-14"),
        clip_pretrained_tag=cfg["model"].get("clip_pretrained_tag", None),
        clip_ckpt_path=None,
        novelty_cfg=cfg.get("novelty", {}),
    ).to(device)

    # Load FULL checkpoint + EMA
    mm, ema_shadow, ckpt_epoch_abs = load_full_checkpoint(mm_model, args.full_ckpt, device)
    restore_ema_weights(mm, ema_shadow)

    # Validation loader
    val_loader = build_val_loader(cfg)

    # Load prototypes (image + pair)
    proto_ckpt = torch.load(args.proto_path, map_location=device)

    if "prototypes_img" in proto_ckpt:
        # New format from updated compute_mm_prototypes.py
        protos_img = proto_ckpt["prototypes_img"].to(device)  # [2, D]
        protos_pair = proto_ckpt.get("prototypes_pair", None)
        if protos_pair is not None:
            protos_pair = protos_pair.to(device)  # [2, 4D]
    else:
        # Backwards compatibility: old file had just "prototypes" (image-only)
        protos_img = proto_ckpt["prototypes"].to(device)
        protos_pair = None

    # Image prototypes: 0 = real, 1 = fake
    p_img_real = F.normalize(protos_img[0], dim=-1)
    p_img_fake = F.normalize(protos_img[1], dim=-1)

    # Pair prototypes: 0 = match, 1 = mismatch (if available)
    if protos_pair is not None:
        p_pair_match = F.normalize(protos_pair[0], dim=-1)
        p_pair_mismatch = F.normalize(protos_pair[1], dim=-1)
    else:
        p_pair_match = None
        p_pair_mismatch = None

    # --- scoring functions ----------------------------------------

    def score_img_only(img_z: torch.Tensor, txt_z: torch.Tensor | None = None) -> torch.Tensor:
        """Image-only prototype score: higher = more fake."""
        z_img = F.normalize(img_z, dim=-1)
        s_real = (z_img * p_img_real).sum(dim=-1)
        s_fake = (z_img * p_img_fake).sum(dim=-1)
        return s_fake - s_real

    def score_pair_only(img_z: torch.Tensor, txt_z: torch.Tensor) -> torch.Tensor:
        """
        Pair prototype score in [img, txt, |img-txt|, img*txt] space.
        0 = match, 1 = mismatch. Returns logit (higher = more fake / mismatch).
        """
        if p_pair_match is None or p_pair_mismatch is None:
            raise RuntimeError(
                "Pair prototypes not found in proto_ckpt, but pair-mode scores were requested."
            )

        z_img = F.normalize(img_z, dim=-1)
        z_txt = F.normalize(txt_z, dim=-1)

        diff = (z_img - z_txt).abs()
        prod = z_img * z_txt
        pair_feat = torch.cat([z_img, z_txt, diff, prod], dim=-1)  # [B, 4D]
        pair_feat = F.normalize(pair_feat, dim=-1)

        s_match = (pair_feat * p_pair_match).sum(dim=-1)
        s_mismatch = (pair_feat * p_pair_mismatch).sum(dim=-1)
        return s_mismatch - s_match

    score_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "img_only": score_img_only
    }
    if p_pair_match is not None and p_pair_mismatch is not None:
        score_fns["pair"] = score_pair_only

    print(f"[DET] Evaluating modes: {list(score_fns.keys())}", flush=True)

    epoch_abs = args.epoch_abs if args.epoch_abs > 0 else (ckpt_epoch_abs if ckpt_epoch_abs is not None else -1)
    epoch_rel = args.epoch_rel if args.epoch_rel > 0 else -1
    tpl_eval = cfg["model"].get("clip_prompt_templates_eval", ["{caption}"])

    max_adv_batches = int(cfg.get("eval_adv", {}).get("max_val_batches", 100))

    # === Detector evaluation (clean + adv) ========================
    det_metrics = eval_detector_clean_adv_multi(
        mm,
        val_loader,
        device,
        cfg,
        tpl_eval=tpl_eval,
        score_fns=score_fns,
        max_batches=max_adv_batches,
        fixed_theta=args.fixed_theta,
    )

    for mode_name, m in det_metrics.items():
        print(
            "[DetSummary] "
            f"mode={mode_name} "
            f"clean_acc={m['det_acc_clean']:.3f} "
            f"adv_acc={m['det_acc_adv']:.3f} "
            f"clean_auc={m['det_auc_clean']:.3f} "
            f"adv_auc={m['det_auc_adv']:.3f} "
            f"fool_rate={m['det_fool_rate']:.3f} "
            f"flip_rate={m['det_flip_rate']:.3f}",
            flush=True,
        )

    # Optional W&B logging
    if WANDB_AVAILABLE and wandb.run is not None:
        log_payload: Dict[str, float | int] = {
            "epoch_abs": epoch_abs,
            "epoch_rel": epoch_rel,
        }
        for mode_name, m in det_metrics.items():
            for k, v in m.items():
                log_payload[f"{mode_name}/{k}"] = v
        wandb.log(log_payload)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    print("[MAIN] starting eval_adv_only.py", flush=True)
    try:
        main()
        print("[MAIN] finished eval_adv_only.py", flush=True)
    except Exception:
        import traceback

        print("[MAIN][ERROR] Exception during eval_adv_only:", flush=True)
        traceback.print_exc()
        raise

