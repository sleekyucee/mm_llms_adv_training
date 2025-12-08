#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adversarial evaluation for classifier heads (img_only, i2t, t2i)
trained on top of a frozen, adversarially trained multimodal backbone.

- Loads MMHeadsClassifier + backbone from checkpoint
- Runs PGD on images for each mode
- Reports, per mode:
    * clean_acc
    * adv_acc
    * fool_rate  (clean-correct -> adv-incorrect)
    * flip_rate  (prediction changed clean -> adv)
    * clean_auc
    * adv_auc
- Optionally logs metrics to Weights & Biases.
"""

import timm  # keep near top

# ------------------------------------------------------------------
# Disable timm pretrained downloads on HPC
# ------------------------------------------------------------------
_real_create_model = timm.create_model


def _create_model_offline(*args, **kwargs):
    kwargs["pretrained"] = False
    return _real_create_model(*args, **kwargs)


timm.create_model = _create_model_offline

# ------------------------------------------------------------------
# Standard imports
# ------------------------------------------------------------------
import argparse
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- W&B -----------------------------------------------------------
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False
# ------------------------------------------------------------------

from utils_mm import load_config, seed_all
from dataset_mm import FFPPTripletDataset
from model_mm import MMModel
from train_contrastive import encode_with_prompt_ensemble
from train_classifier import MMHeadsClassifier, collate_mm_classifier, pick_one_caption


# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def build_mm_backbone(cfg: Dict[str, Any], device: torch.device) -> MMModel:
    """
    Rebuild MMModel with the same config used for contrastive training.
    We only need architecture; we then load classifier weights which
    already contain the backbone parameters (they were saved together).
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


def logits_for_mode_with_grad(
    clf_model: MMHeadsClassifier,
    imgs: torch.Tensor,
    texts: List[str],
    templates: List[str],
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    """
    Compute logits for a given mode without torch.no_grad(), so that
    gradients flow back to imgs for PGD.
    """
    mm = clf_model.mm  # frozen backbone
    mm.eval()          # eval mode, but grads allowed

    imgs = imgs.to(device)

    if mode == "img_only":
        img_z = mm.encode_image(imgs)
        img_z = F.normalize(img_z, dim=-1)
        return clf_model.head_img(img_z)

    # i2t / t2i: image + text
    img_z = mm.encode_image(imgs)
    img_z = F.normalize(img_z, dim=-1)

    txt_z = encode_with_prompt_ensemble(mm, texts, templates, device)
    txt_z = F.normalize(txt_z, dim=-1)

    # optional refiner (FiLM + ForensicToken) – but without no_grad
    refiner = getattr(mm, "refine_embeddings", None)
    if callable(refiner):
        img_z, txt_z = refiner(img_z, txt_z, imgs)
        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)

    diff = torch.abs(img_z - txt_z)
    prod = img_z * txt_z
    pair_feat = torch.cat([img_z, txt_z, diff, prod], dim=-1)

    if mode == "i2t":
        logits = clf_model.head_i2t(pair_feat)
    elif mode == "t2i":
        logits = clf_model.head_t2i(pair_feat)
    else:
        raise ValueError(f"Unknown mode for PGD: {mode}")

    return logits


def pgd_attack_images(
    imgs: torch.Tensor,
    labels: torch.Tensor,
    model: MMHeadsClassifier,
    texts: List[str],
    templates: List[str],
    device: torch.device,
    mode: str,
    eps: float,
    step_size: float,
    steps: int,
    rand_start: bool = True,
) -> torch.Tensor:
    """
    Standard L_inf PGD on images for a single head (mode),
    using a gradient-enabled forward path.
    """
    model.eval()
    imgs = imgs.to(device)
    labels = labels.to(device)

    if rand_start:
        delta = torch.empty_like(imgs).uniform_(-eps, eps)
        x_adv = (imgs + delta).clamp(0.0, 1.0)
    else:
        x_adv = imgs.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        if x_adv.grad is not None:
            x_adv.grad = None

        logits = logits_for_mode_with_grad(
            model,
            x_adv,
            texts,
            templates,
            device,
            mode=mode,
        )
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + step_size * grad_sign
            x_adv = torch.max(torch.min(x_adv, imgs + eps), imgs - eps)
            x_adv = x_adv.clamp(0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv.detach()


def eval_adv_mm_heads(
    clf_model: MMHeadsClassifier,
    loader: DataLoader,
    device: torch.device,
    templates: List[str],
    eps: float = 8.0 / 255.0,
    step_size: float = 2.0 / 255.0,
    steps: int = 5,
    max_batches: int | None = None,
    amp_mode: str = "bf16",
    modes: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate robustness per mode with fool rate and flip rate tracking.
    """
    if modes is None:
        modes = ["img_only", "i2t", "t2i"]

    clf_model.eval()

    # Global stats per mode
    stats = {
        m: {
            "total": 0,
            "clean_correct": 0,
            "adv_correct": 0,
            "flip_count": 0,
            "clean_labels": [],
            "clean_probs": [],
            "adv_probs": [],
        }
        for m in modes
    }

    # Per-manip stats per mode
    per_manip: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in modes}

    use_amp = amp_mode in ["fp16", "bf16"]
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16

    for bi, batch in enumerate(
        tqdm(loader, desc=f"AdvEval (PGD {steps} steps)", leave=False)
    ):
        if max_batches is not None and bi >= max_batches:
            break

        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        texts_batch = pick_one_caption(batch["texts"])
        B = labels.size(0)

        # --------------------------------------------------
        # manipulation-type detection
        # --------------------------------------------------
        manip_field = None
        if "manip_type" in batch:
            manip_field = batch["manip_type"]
        elif "manip" in batch:
            manip_field = batch["manip"]
        elif "source" in batch:
            manip_field = batch["source"]

        manip_list = None

        if manip_field is not None:
            manip_list = list(manip_field)
        else:
            paths = None
            if "image_rel" in batch:
                paths = batch["image_rel"]
            elif "image_path" in batch:
                paths = batch["image_path"]

            if paths is not None:
                tmp_list = []
                for p in paths:
                    mt = "unknown"
                    p_str = str(p)
                    p_low = p_str.lower()

                    if p_low.startswith("real/") or "/real/" in p_low:
                        mt = "real"
                    elif p_low.startswith("fake/") or "/fake/" in p_low:
                        mt = "deepfake"
                    elif "dfd/imgs" in p_str or "/dfd/" in p_str:
                        mt = "dfd"
                    elif "dfdc/imgs" in p_str or "/dfdc/" in p_str:
                        mt = "dfdc"
                    elif "dpf/imgs" in p_str or "/dpf/" in p_str:
                        mt = "dpf"
                    elif "mffdi/imgs" in p_str or "/mffdi/" in p_str:
                        mt = "mffdi"
                    elif "pgc/imgs" in p_str or "/pgc/" in p_str:
                        mt = "pgc"
                    elif "wfir/imgs" in p_str or "/wfir/" in p_str:
                        mt = "wfir"

                    tmp_list.append(mt)

                manip_list = tmp_list

        for mode in modes:
            st = stats[mode]
            st["total"] += B

            # -------- CLEAN PASS --------
            with torch.no_grad():
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits_clean = clf_model(imgs, texts_batch, templates, device, mode=mode)
                    probs_clean = logits_clean.softmax(dim=1)[:, 1]
                    pred_clean = logits_clean.argmax(dim=1)

            clean_correct = (pred_clean == labels).sum().item()
            st["clean_correct"] += clean_correct

            labels_cpu = labels.detach().cpu()
            p_clean_cpu = probs_clean.detach().cpu()

            st["clean_labels"].append(labels_cpu)
            st["clean_probs"].append(p_clean_cpu)

            # -------- ADVERSARIAL PASS --------
            imgs_adv = pgd_attack_images(
                imgs=imgs,
                labels=labels,
                model=clf_model,
                texts=texts_batch,
                templates=templates,
                device=device,
                mode=mode,
                eps=eps,
                step_size=step_size,
                steps=steps,
                rand_start=True,
            )

            with torch.no_grad():
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits_adv = clf_model(imgs_adv, texts_batch, templates, device, mode=mode)
                    probs_adv = logits_adv.softmax(dim=1)[:, 1]
                    pred_adv = logits_adv.argmax(dim=1)

            adv_correct = (pred_adv == labels).sum().item()
            st["adv_correct"] += adv_correct

            # Flip count: prediction changed, regardless of correctness
            flip_count = (pred_clean != pred_adv).sum().item()
            st["flip_count"] += flip_count

            # Fool count: clean correct -> adv incorrect
            clean_correct_mask = pred_clean == labels
            adv_incorrect_mask = pred_adv != labels
            fool_mask = clean_correct_mask & adv_incorrect_mask
            num_fool = fool_mask.sum().item()

            st.setdefault("fool_count", 0)
            st["fool_count"] += num_fool

            p_adv_cpu = probs_adv.detach().cpu()
            st["adv_probs"].append(p_adv_cpu)

            # --------- Per-manip accumulation ----------#
            if manip_list is not None:
                for j in range(B):
                    mt = manip_list[j]
                    if isinstance(mt, bytes):
                        mt = mt.decode("utf-8", errors="ignore")
                    mt = str(mt)

                    pm_mode = per_manip[mode].setdefault(
                        mt,
                        {
                            "total": 0,
                            "clean_correct": 0,
                            "adv_correct": 0,
                            "flip_count": 0,
                            "fool_count": 0,
                            "clean_labels": [],
                            "clean_probs": [],
                            "adv_probs": [],
                        },
                    )

                    pm_mode["total"] += 1

                    lbl_j = labels_cpu[j : j + 1]
                    pc_j = p_clean_cpu[j : j + 1]
                    pa_j = p_adv_cpu[j : j + 1]

                    if pred_clean[j] == labels[j]:
                        pm_mode["clean_correct"] += 1
                    if pred_adv[j] == labels[j]:
                        pm_mode["adv_correct"] += 1

                    if pred_clean[j] != pred_adv[j]:
                        pm_mode["flip_count"] += 1

                    if (pred_clean[j] == labels[j]) and (pred_adv[j] != labels[j]):
                        pm_mode["fool_count"] += 1

                    pm_mode["clean_labels"].append(lbl_j)
                    pm_mode["clean_probs"].append(pc_j)
                    pm_mode["adv_probs"].append(pa_j)

    # -------- Aggregate per-mode metrics (global) --------
    out: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        st = stats[mode]
        total = max(1, st["total"])

        clean_acc = st["clean_correct"] / total
        adv_acc = st["adv_correct"] / total

        # Fool rate: clean correct -> adv incorrect
        fool_den = max(1, st["clean_correct"])
        fool_rate = st.get("fool_count", 0) / fool_den

        # Flip rate: predictions changed (regardless of correctness)
        flip_rate = st.get("flip_count", 0) / total

        y_clean = torch.cat(st["clean_labels"]).numpy()
        p_clean = torch.cat(st["clean_probs"]).numpy()
        p_adv = torch.cat(st["adv_probs"]).numpy()

        clean_auc = safe_auc(y_clean, p_clean)
        adv_auc = safe_auc(y_clean, p_adv)

        out[mode] = {
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "fool_rate": fool_rate,
            "flip_rate": flip_rate,
            "clean_auc": clean_auc,
            "adv_auc": adv_auc,
        }

    # -------- Per-manip printout --------
    for mode in modes:
        pm = per_manip[mode]
        if not pm:
            continue

        print(f"\n[Per-manip breakdown for mode={mode}]", flush=True)
        for mt, st in pm.items():
            total = max(1, st["total"])
            clean_acc = st["clean_correct"] / total
            adv_acc = st["adv_correct"] / total

            fool_den = max(1, st["clean_correct"])
            fool_rate = st.get("fool_count", 0) / fool_den
            flip_rate = st.get("flip_count", 0) / total

            try:
                y_clean = torch.cat(st["clean_labels"]).numpy()
                p_clean = torch.cat(st["clean_probs"]).numpy()
                p_adv = torch.cat(st["adv_probs"]).numpy()
                clean_auc = safe_auc(y_clean, p_clean)
                adv_auc = safe_auc(y_clean, p_adv)
            except Exception:
                clean_auc = float("nan")
                adv_auc = float("nan")

            print(
                f"  - manip={mt:20s} "
                f"clean_acc={clean_acc:.3f} "
                f"adv_acc={adv_acc:.3f} "
                f"fool_rate={fool_rate:.3f} "
                f"flip_rate={flip_rate:.3f} "
                f"clean_auc={clean_auc:.3f} "
                f"adv_auc={adv_auc:.3f}",
                flush=True,
            )

    return out


# ------------------------------------------------------------------
# Argparse / main
# ------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (same as training)")
    ap.add_argument("--clf_ckpt", required=True, help="Path to mm_heads_best.pt or FULL ckpt")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- W&B init ---------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = WANDB_AVAILABLE and wandb_cfg.get("enable", False)
    if use_wandb:
        run_name = wandb_cfg.get("eval_run_name", "mm_heads_adv_eval")
        project = wandb_cfg.get("project", "mm_forensics")
        entity = wandb_cfg.get("entity", None)
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=cfg,
            job_type="adv_eval",
            mode="offline",
        )
    # --------------------------------------------------------------

    # ---- rebuild backbone + classifier and load checkpoint -------
    mm = build_mm_backbone(cfg, device)
    embed_dim = cfg.get("clf", {}).get("embed_dim", 1024)

    clf_model = MMHeadsClassifier(
        mm,
        embed_dim=embed_dim,
        share_pair_head=False,
    ).to(device)

    ckpt = torch.load(args.clf_ckpt, map_location="cpu")

    if isinstance(ckpt, dict):
        if "classifier_state_dict" in ckpt:
            # thin checkpoint: mm_heads_best.pt
            state_dict = ckpt["classifier_state_dict"]
        elif "model" in ckpt:
            # full checkpoint: mm_heads_BEST_FULL.pt or mm_heads_LAST_FULL.pt
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = clf_model.load_state_dict(state_dict, strict=False)
    print(f"[WARN] Missing keys: {missing}")
    print(f"[WARN] Unexpected keys: {unexpected}")

    clf_model.eval()
    # --------------------------------------------------------------

    # ---- data (use val or test manifest) -------------------------
    data_cfg = cfg["data"]
    project_root = data_cfg.get("project_root", ".")
    image_size = data_cfg.get("image_size", 224)

    # frames_root can be omitted / null for cross-datasets with full paths
    frames_root = data_cfg.get("frames_root", None)
    if frames_root is None:
        frames_root = project_root

    val_ds = FFPPTripletDataset(
        jsonl_path=data_cfg["val_manifest"],   # swap to test manifest when needed
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
    # --------------------------------------------------------------

    adv_cfg = cfg.get("clf_adv", {})
    eps = adv_cfg.get("eps", 8.0 / 255.0)
    step_size = adv_cfg.get("step_size", 2.0 / 255.0)
    steps = adv_cfg.get("steps", 5)
    max_batches = adv_cfg.get("max_batches", None)
    amp_mode = adv_cfg.get("amp", "bf16")

    templates = cfg["model"].get("clip_prompt_templates_eval", ["{caption}"])

    # modes
    clf_eval_cfg = cfg.get("clf_eval", {})
    modes = clf_eval_cfg.get("modes", ["img_only", "i2t", "t2i"])

    stats = eval_adv_mm_heads(
        clf_model=clf_model,
        loader=val_loader,
        device=device,
        templates=templates,
        eps=eps,
        step_size=step_size,
        steps=steps,
        max_batches=max_batches,
        amp_mode=amp_mode,
        modes=modes,
    )

    # ---- print + wandb log ---------------------------------------
    log_dict = {}
    for mode, st in stats.items():
        print(
            f"[AdvEval-{mode}] "
            f"clean_acc={st['clean_acc']:.3f} "
            f"adv_acc={st['adv_acc']:.3f} "
            f"fool_rate={st['fool_rate']:.3f} "
            f"flip_rate={st['flip_rate']:.3f} "
            f"clean_auc={st['clean_auc']:.3f} "
            f"adv_auc={st['adv_auc']:.3f}",
            flush=True,
        )
        prefix = f"adv/{mode}"
        log_dict.update(
            {
                f"{prefix}/clean_acc": st["clean_acc"],
                f"{prefix}/adv_acc": st["adv_acc"],
                f"{prefix}/fool_rate": st["fool_rate"],
                f"{prefix}/flip_rate": st["flip_rate"],
                f"{prefix}/clean_auc": st["clean_auc"],
                f"{prefix}/adv_auc": st["adv_auc"],
            }
        )

    if use_wandb:
        wandb.log(log_dict)
        wandb.finish()


if __name__ == "__main__":
    main()

