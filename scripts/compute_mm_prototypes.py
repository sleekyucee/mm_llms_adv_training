# scripts/compute_mm_prototypes.py
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from utils_mm import load_config, seed_all
from dataset_mm import FFPPTripletDataset
from model_mm import MMModel
from train_contrastive import (
    collate_mm,
    encode_with_prompt_ensemble,
    build_text_batch_for_it_balanced,
)

@torch.no_grad()
def compute_prototypes(
    mm: MMModel,
    loader: DataLoader,
    device: torch.device,
    tpl_train,
    num_classes: int = 2,
    amp_mode: str = "off",
):
    """
    Compute:
      - image-only class prototypes:  shape [num_classes, D_img]
      - pair prototypes (match/mismatch): shape [2, D_pair=4*D_img]

    Pair labels:
      y_pair = 0  -> "match"   (label == 0, real / genuine)
      y_pair = 1  -> "mismatch" (label == 1, fake / manipulated)
    """
    amp_enabled = amp_mode in ("fp16", "bf16")
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16

    dim_img = None
    dim_pair = None

    sum_img = None               # [num_classes, D_img]
    counts_img = torch.zeros(num_classes, device=device, dtype=torch.long)

    sum_pair = None              # [2, D_pair]  (0=match, 1=mismatch)
    counts_pair = torch.zeros(2, device=device, dtype=torch.long)

    total_steps = len(loader)
    print(f"[PROTO] starting accumulation over {total_steps} batches")

    for step, batch in enumerate(loader):
        if step % 50 == 0:
            print(f"[PROTO] batch {step}/{total_steps}")

        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)   # 0=real,1=fake
        texts_list = batch["texts"]

        picked = build_text_batch_for_it_balanced(
            texts_list,
            step=step,
        )

        with autocast(enabled=amp_enabled, dtype=amp_dtype):
            img_z_raw = mm.encode_image(imgs)
            txt_z_raw = encode_with_prompt_ensemble(mm, picked, tpl_train, device)

            refiner = getattr(mm, "refine_embeddings", None)
            if callable(refiner):
                img_z, txt_z = refiner(img_z_raw, txt_z_raw, imgs)
            else:
                img_z, txt_z = img_z_raw, txt_z_raw

            img_z = F.normalize(img_z, dim=-1)
            txt_z = F.normalize(txt_z, dim=-1)

        # -------------------------
        # Image-only prototypes
        # -------------------------
        if dim_img is None:
            dim_img = img_z.size(-1)
            sum_img = torch.zeros(num_classes, dim_img, device=device)

        for c in range(num_classes):
            mask = (labels == c)
            if mask.any():
                sum_img[c] += img_z[mask].sum(dim=0)
                counts_img[c] += mask.sum()

        # -------------------------
        # Pair prototypes (match / mismatch)
        #    match    -> label == 0 (real)
        #    mismatch -> label == 1 (fake)
        # -------------------------
        # Build pair feature: [img, txt, |img-txt|, img*txt]
        diff = (img_z - txt_z).abs()
        prod = img_z * txt_z
        pair_feat = torch.cat([img_z, txt_z, diff, prod], dim=-1)   # [B, 4D]

        if dim_pair is None:
            dim_pair = pair_feat.size(-1)
            sum_pair = torch.zeros(2, dim_pair, device=device)

        # y_pair: 0 = match (real), 1 = mismatch (fake)
        y_pair = labels.clamp(0, 1)  # ensure 0/1

        for c_pair in (0, 1):
            mask_p = (y_pair == c_pair)
            if mask_p.any():
                sum_pair[c_pair] += pair_feat[mask_p].sum(dim=0)
                counts_pair[c_pair] += mask_p.sum()

    # Finalise prototypes
    protos_img = sum_img / counts_img.clamp_min(1).unsqueeze(-1)
    protos_img = F.normalize(protos_img, dim=-1)

    protos_pair = sum_pair / counts_pair.clamp_min(1).unsqueeze(-1)
    protos_pair = F.normalize(protos_pair, dim=-1)

    print(f"[PROTO] done.")
    print(f"[PROTO] image counts per class: {counts_img.tolist()}")
    print(f"[PROTO] pair counts match/mismatch: {counts_pair.tolist()}")

    return protos_img, counts_img, protos_pair, counts_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init-from", type=str, required=True,
                        help="Path to FINAL robust backbone checkpoint (.pt)")
    parser.add_argument("--out", type=str, default="outputs/forgery_prototypes.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === dataset / loader: TRAIN SPLIT only ===
    model_type = cfg["model"].get("type", "clip")
    clip_norm = (model_type == "clip")

    train_ds = FFPPTripletDataset(
        cfg["data"]["train_manifest"], cfg["data"]["frames_root"],
        image_size=cfg["data"].get("image_size", 224),
        augment=False,
        clip_norm=clip_norm,
        return_pair=False,
    )

    numw = min(cfg["train"].get("num_workers", 4), 8)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=numw,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_mm,
    )

    # === model ===
    model = MMModel(
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
        clip_ckpt_path=cfg["model"].get("clip_ckpt_path", None),
        novelty_cfg=cfg.get("novelty", {}),
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    mm = model.module if isinstance(model, torch.nn.DataParallel) else model

    # load FINAL checkpoint
    print(f"[PROTO] loading weights from {args.init_from}")
    ckpt = torch.load(args.init_from, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    mm.load_state_dict(sd, strict=False)
    mm.eval()

    amp_mode = cfg["train"].get("amp", "off")
    tpl_train = cfg["model"].get("clip_prompt_templates_train", ["{caption}"])

    protos_img, counts_img, protos_pair, counts_pair = compute_prototypes(
        mm,
        train_loader,
        device,
        tpl_train=tpl_train,
        num_classes=int(cfg.get("loss", {}).get("forgery_num_classes", 2)),
        amp_mode=amp_mode,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "prototypes_img": protos_img.cpu(),
            "counts_img": counts_img.cpu(),
            "prototypes_pair": protos_pair.cpu(),
            "counts_pair": counts_pair.cpu(),
            "config": cfg,
        },
        args.out,
    )
    print(f"[PROTO] saved to {args.out}")
    print(f"[PROTO] image counts per class: {counts_img.tolist()}")
    print(f"[PROTO] pair counts match/mismatch: {counts_pair.tolist()}")


if __name__ == "__main__":
    main()
