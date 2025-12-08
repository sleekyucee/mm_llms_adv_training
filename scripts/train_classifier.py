#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Dict, Any

import timm

_real_create_model = timm.create_model


def _create_model_offline(*args, **kwargs):
    kwargs["pretrained"] = False
    return _real_create_model(*args, **kwargs)


timm.create_model = _create_model_offline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from utils_mm import load_config, seed_all
from dataset_mm import FFPPTripletDataset
from model_mm import MMModel
from train_contrastive import encode_with_prompt_ensemble, _strong_text_perturb


class MMHeadsClassifier(nn.Module):
    def __init__(self, mm_backbone: MMModel, embed_dim: int = 1024, share_pair_head: bool = False):
        super().__init__()
        self.mm = mm_backbone
        self.embed_dim = embed_dim

        for p in self.mm.parameters():
            p.requires_grad_(False)

        self.head_img = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 2),
        )

        pair_dim = self.embed_dim * 4
        pair_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 2),
        )

        self.head_i2t = pair_head
        self.head_t2i = (
            pair_head
            if share_pair_head
            else nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, 2),
            )
        )

    def _encode_img_txt(self, imgs: torch.Tensor, texts: List[str], templates: List[str], device: torch.device):
        self.mm.eval()
        img_z_raw = self.mm.encode_image(imgs)
        img_z_raw = F.normalize(img_z_raw, dim=-1)

        txt_z_raw = encode_with_prompt_ensemble(self.mm, texts, templates, device)
        txt_z_raw = F.normalize(txt_z_raw, dim=-1)

        refiner = getattr(self.mm, "refine_embeddings", None)
        if callable(refiner):
            img_z, txt_z = refiner(img_z_raw, txt_z_raw, imgs)
            img_z = F.normalize(img_z, dim=-1)
            txt_z = F.normalize(txt_z, dim=-1)
        else:
            img_z, txt_z = img_z_raw, txt_z_raw

        return img_z, txt_z

    def _make_pair_features(self, img_z: torch.Tensor, txt_z: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(img_z - txt_z)
        prod = img_z * txt_z
        return torch.cat([img_z, txt_z, diff, prod], dim=-1)

    def forward(
        self,
        imgs: torch.Tensor,
        texts: List[str],
        templates: List[str],
        device: torch.device,
        mode: str = "img_only",
    ) -> torch.Tensor:
        if mode == "img_only":
            self.mm.eval()
            img_z = self.mm.encode_image(imgs)
            img_z = F.normalize(img_z, dim=-1)

            if img_z.shape[-1] != self.embed_dim:
                raise RuntimeError(f"img_only: expected embed_dim={self.embed_dim}, got {img_z.shape[-1]}")
            return self.head_img(img_z)

        img_z, txt_z = self._encode_img_txt(imgs, texts, templates, device)
        if img_z.shape[-1] != self.embed_dim:
            raise RuntimeError(f"pair: expected embed_dim={self.embed_dim}, got {img_z.shape[-1]}")

        pair_feat = self._make_pair_features(img_z, txt_z)

        if mode == "i2t":
            return self.head_i2t(pair_feat)
        elif mode == "t2i":
            return self.head_t2i(pair_feat)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def collate_mm_classifier(batch):
    import torch

    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    texts = [b["texts"] for b in batch]

    out = {
        "image": images,
        "label": labels,
        "texts": texts,
    }

    if "blend_img" in batch[0]:
        blend_imgs = [b["blend_img"] for b in batch]
        if isinstance(blend_imgs[0], torch.Tensor):
            out["blend_img"] = torch.stack(blend_imgs, dim=0)
        else:
            out["blend_img"] = blend_imgs

    for key in ["image_rel", "image_path", "manip_type", "manip", "source"]:
        if key in batch[0]:
            out[key] = [b[key] for b in batch]

    return out


def pick_one_caption(texts_list: List[List[str]]) -> List[str]:
    out: List[str] = []
    for ts in texts_list:
        if not ts:
            out.append("")
        else:
            out.append(ts[0])
    return out


def train_one_epoch(
    model: MMHeadsClassifier,
    loader: DataLoader,
    device: torch.device,
    templates: List[str],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    amp_mode: str = "off",
    epoch: int = 1,
    num_epochs: int = 1,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_img = correct_i2t = correct_t2i = 0

    use_amp = amp_mode in ["fp16", "bf16"]
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16

    for batch in tqdm(loader, desc=f"Train {epoch}/{num_epochs}", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        texts_batch = pick_one_caption(batch["texts"])

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits_img = model(imgs, texts_batch, templates, device, mode="img_only")
            logits_i2t = model(imgs, texts_batch, templates, device, mode="i2t")
            logits_t2i = model(imgs, texts_batch, templates, device, mode="t2i")

            loss = ce(logits_img, labels) + ce(logits_i2t, labels) + ce(logits_t2i, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        B = labels.size(0)
        total_loss += loss.item() * B
        total_samples += B

        correct_img += (logits_img.argmax(dim=1) == labels).sum().item()
        correct_i2t += (logits_i2t.argmax(dim=1) == labels).sum().item()
        correct_t2i += (logits_t2i.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / max(1, total_samples)
    return {
        "loss": avg_loss,
        "acc_img": correct_img / max(1, total_samples),
        "acc_i2t": correct_i2t / max(1, total_samples),
        "acc_t2i": correct_t2i / max(1, total_samples),
    }


def pgd_attack_images_classifier(
    model,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    texts: list[str],
    templates: list[str],
    device: torch.device,
    mode: str,
    eps: float,
    step_size: float,
    steps: int,
    rand_start: bool = True,
) -> torch.Tensor:
    model.eval()
    imgs = imgs.to(device)
    labels = labels.to(device)

    x_adv = imgs.detach().clone()
    if rand_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        logits = model(x_adv, texts, templates, device, mode=mode)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        if x_adv.grad is None:
            raise RuntimeError("x_adv.grad is None during PGD")

        grad_sign = x_adv.grad.sign()
        x_adv = x_adv.detach() + step_size * grad_sign
        x_adv = torch.max(torch.min(x_adv, imgs + eps), imgs - eps)
        x_adv = x_adv.clamp(0.0, 1.0)

    model.train()
    return x_adv.detach()


def train_one_epoch_adv(
    model: MMHeadsClassifier,
    loader: DataLoader,
    device: torch.device,
    templates: list[str],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    amp_mode: str = "off",
    epoch: int = 1,
    num_epochs: int = 1,
    adv_eps: float = 0.031,
    adv_step_size: float = 0.0075,
    adv_steps: int = 10,
    text_prob: float = 0.7,
    text_max_subs: int = 4,
    text_mismatch_prob: float = 0.4,
    max_adv_batches: int | None = None,
) -> dict:
    model.train()

    def balanced_ce_loss(logits, labels, alpha=0.9):
        batch_size = labels.size(0)
        real_count = (labels == 0).sum().item()
        fake_count = (labels == 1).sum().item()

        if real_count == 0 or fake_count == 0:
            return F.cross_entropy(logits, labels)

        weight_real = 20.0 / (real_count + 1e-8)
        weight_fake = 1.0 / (fake_count + 1e-8)

        weights = torch.tensor([weight_real, weight_fake], device=logits.device)
        weights = weights / weights.sum() * 2

        return F.cross_entropy(logits, labels, weight=weights)

    def adversarial_text_perturbation(
        model,
        imgs,
        texts,
        templates,
        device,
        labels,
        text_prob=0.7,
        max_subs=4,
        mismatch_prob=0.4,
        num_candidates=3,
    ):
        model.eval()
        worst_texts = texts.copy()

        for i in range(len(texts)):
            current_text = texts[i]
            best_text = current_text
            max_loss = -float("inf")

            candidates = [current_text]
            for _ in range(num_candidates - 1):
                candidates.append(
                    _strong_text_perturb(
                        [current_text],
                        prob=text_prob,
                        max_subs=max_subs,
                        mismatch_prob=mismatch_prob,
                    )[0]
                )

            for cand_text in candidates:
                with torch.no_grad():
                    logits_i2t = model(imgs[i : i + 1], [cand_text], templates, device, "i2t")
                    logits_t2i = model(imgs[i : i + 1], [cand_text], templates, device, "t2i")

                    loss_i2t = F.cross_entropy(logits_i2t, labels[i : i + 1])
                    loss_t2i = F.cross_entropy(logits_t2i, labels[i : i + 1])
                    total_loss = (loss_i2t + loss_t2i) / 2.0

                if total_loss > max_loss:
                    max_loss = total_loss
                    best_text = cand_text

            worst_texts[i] = best_text

        model.train()
        return worst_texts

    total_loss = 0.0
    total_samples = 0
    correct_img = correct_i2t = correct_t2i = 0

    flip_rates = {"img": [], "i2t_img": [], "i2t_txt": [], "t2i_img": [], "t2i_txt": []}
    flip_rates_real = {k: [] for k in flip_rates.keys()}
    flip_rates_fake = {k: [] for k in flip_rates.keys()}

    use_amp = amp_mode in ["fp16", "bf16"]
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16

    lambda_adv_weight = min(1.0, epoch / 5.0)
    lambda_cons = 0.3
    lambda_balance = 0.5

    for bi, batch in enumerate(tqdm(loader, desc=f"TrainAdv {epoch}/{num_epochs}", leave=False)):
        if max_adv_batches is not None and bi >= max_adv_batches:
            break

        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        texts_batch = pick_one_caption(batch["texts"])
        B = labels.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits_img_clean = model(imgs, texts_batch, templates, device, "img_only")
            logits_i2t_clean = model(imgs, texts_batch, templates, device, "i2t")
            logits_t2i_clean = model(imgs, texts_batch, templates, device, "t2i")

            loss_clean_img = balanced_ce_loss(logits_img_clean, labels, alpha=0.9)
            loss_clean_i2t = balanced_ce_loss(logits_i2t_clean, labels, alpha=0.9)
            loss_clean_t2i = balanced_ce_loss(logits_t2i_clean, labels, alpha=0.9)

            loss_clean = (loss_clean_img + loss_clean_i2t + loss_clean_t2i) / 3.0

        with torch.no_grad():
            pred_img_clean = logits_img_clean.argmax(dim=1)
            pred_i2t_clean = logits_i2t_clean.argmax(dim=1)
            pred_t2i_clean = logits_t2i_clean.argmax(dim=1)

            real_mask = labels == 0
            fake_mask = labels == 1

        model.eval()
        imgs_adv = imgs.clone()

        if real_mask.any():
            real_imgs = imgs[real_mask]
            real_labels = labels[real_mask]
            real_texts = [texts_batch[i] for i in range(len(texts_batch)) if real_mask[i]]

            real_adv = pgd_attack_images_classifier(
                model=model,
                imgs=real_imgs,
                labels=real_labels,
                texts=real_texts,
                templates=templates,
                device=device,
                mode="img_only",
                eps=adv_eps * 0.1,
                step_size=adv_step_size * 0.1,
                steps=max(3, adv_steps // 2),
                rand_start=True,
            )
            imgs_adv[real_mask] = real_adv

        if fake_mask.any():
            fake_imgs = imgs[fake_mask]
            fake_labels = labels[fake_mask]
            fake_texts = [texts_batch[i] for i in range(len(texts_batch)) if fake_mask[i]]

            fake_adv = pgd_attack_images_classifier(
                model=model,
                imgs=fake_imgs,
                labels=fake_labels,
                texts=fake_texts,
                templates=templates,
                device=device,
                mode="img_only",
                eps=adv_eps * 2.0,
                step_size=adv_step_size,
                steps=adv_steps,
                rand_start=True,
            )
            imgs_adv[fake_mask] = fake_adv

        model.train()

        texts_adv = texts_batch.copy()
        if real_mask.any():
            real_texts = [texts_batch[i] for i in range(len(texts_batch)) if real_mask[i]]
            real_texts_adv = adversarial_text_perturbation(
                model=model,
                imgs=imgs[real_mask],
                texts=real_texts,
                templates=templates,
                device=device,
                labels=labels[real_mask],
                text_prob=text_prob,
                max_subs=text_max_subs,
                mismatch_prob=text_mismatch_prob,
                num_candidates=2,
            )

            real_idx = 0
            for i in range(len(texts_batch)):
                if real_mask[i]:
                    texts_adv[i] = real_texts_adv[real_idx]
                    real_idx += 1

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits_img_adv = model(imgs_adv, texts_batch, templates, device, "img_only")
            logits_i2t_img_adv = model(imgs_adv, texts_batch, templates, device, "i2t")
            logits_t2i_img_adv = model(imgs_adv, texts_batch, templates, device, "t2i")

            logits_i2t_txt_adv = model(imgs, texts_adv, templates, device, "i2t")
            logits_t2i_txt_adv = model(imgs, texts_adv, templates, device, "t2i")

            loss_adv_img = balanced_ce_loss(logits_img_adv, labels, alpha=0.9)
            loss_adv_i2t_img = balanced_ce_loss(logits_i2t_img_adv, labels, alpha=0.9)
            loss_adv_t2i_img = balanced_ce_loss(logits_t2i_img_adv, labels, alpha=0.9)
            loss_adv_i2t_txt = balanced_ce_loss(logits_i2t_txt_adv, labels, alpha=0.9)
            loss_adv_t2i_txt = balanced_ce_loss(logits_t2i_txt_adv, labels, alpha=0.9)

            loss_adv = (
                loss_adv_img
                + loss_adv_i2t_img
                + loss_adv_t2i_img
                + loss_adv_i2t_txt
                + loss_adv_t2i_txt
            ) / 5.0

            def consistency_loss(clean_logits, adv_logits):
                p_clean = F.softmax(clean_logits, dim=1)
                p_adv = F.softmax(adv_logits, dim=1)
                kl1 = F.kl_div(p_clean.log(), p_adv, reduction="none").sum(dim=1)
                kl2 = F.kl_div(p_adv.log(), p_clean, reduction="none").sum(dim=1)
                return (kl1 + kl2) / 2.0

            consistency_terms = [
                consistency_loss(logits_img_clean, logits_img_adv),
                consistency_loss(logits_i2t_clean, logits_i2t_img_adv),
                consistency_loss(logits_t2i_clean, logits_t2i_img_adv),
                consistency_loss(logits_i2t_clean, logits_i2t_txt_adv),
                consistency_loss(logits_t2i_clean, logits_t2i_txt_adv),
            ]

            all_consistency = torch.stack(consistency_terms, dim=1)
            consistency_per_sample = all_consistency.mean(dim=1)

            if real_mask.any():
                real_probs_adv = F.softmax(logits_img_adv[real_mask], dim=1)[:, 0]
                real_protection_loss = torch.clamp(1.0 - real_probs_adv, min=0.0).mean()
            else:
                real_protection_loss = torch.tensor(0.0, device=device)

            with torch.no_grad():
                pred_img_adv = logits_img_adv.argmax(dim=1)

                if real_mask.any():
                    real_flip = (
                        pred_img_clean[real_mask] != pred_img_adv[real_mask]
                    ).float().mean()
                else:
                    real_flip = torch.tensor(0.0, device=device)

                if fake_mask.any():
                    fake_flip = (
                        pred_img_clean[fake_mask] != pred_img_adv[fake_mask]
                    ).float().mean()
                else:
                    fake_flip = torch.tensor(0.0, device=device)

                flip_imbalance = torch.abs(real_flip - fake_flip)

            real_flip_penalty = 5.0 * real_flip if real_mask.any() else 0.0
            lambda_real_protection = 3.0

            loss = (
                loss_clean
                + lambda_adv_weight * loss_adv
                + lambda_cons * consistency_per_sample.mean()
                + lambda_balance * flip_imbalance
                + real_flip_penalty
                + lambda_real_protection * real_protection_loss
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

        with torch.no_grad():
            correct_img += (pred_img_clean == labels).sum().item()
            correct_i2t += (pred_i2t_clean == labels).sum().item()
            correct_t2i += (pred_t2i_clean == labels).sum().item()

            pred_img_adv = logits_img_adv.argmax(dim=1)
            pred_i2t_img_adv = logits_i2t_img_adv.argmax(dim=1)
            pred_i2t_txt_adv = logits_i2t_txt_adv.argmax(dim=1)
            pred_t2i_img_adv = logits_t2i_img_adv.argmax(dim=1)
            pred_t2i_txt_adv = logits_t2i_txt_adv.argmax(dim=1)

            flip_rates["img"].append((pred_img_clean != pred_img_adv).float().mean().item())
            flip_rates["i2t_img"].append((pred_i2t_clean != pred_i2t_img_adv).float().mean().item())
            flip_rates["i2t_txt"].append((pred_i2t_clean != pred_i2t_txt_adv).float().mean().item())
            flip_rates["t2i_img"].append((pred_t2i_clean != pred_t2i_img_adv).float().mean().item())
            flip_rates["t2i_txt"].append((pred_t2i_clean != pred_t2i_txt_adv).float().mean().item())

            if real_mask.any():
                flip_rates_real["img"].append(
                    (pred_img_clean[real_mask] != pred_img_adv[real_mask]).float().mean().item()
                )
                flip_rates_real["i2t_img"].append(
                    (pred_i2t_clean[real_mask] != pred_i2t_img_adv[real_mask])
                    .float()
                    .mean()
                    .item()
                )
                flip_rates_real["i2t_txt"].append(
                    (pred_i2t_clean[real_mask] != pred_i2t_txt_adv[real_mask])
                    .float()
                    .mean()
                    .item()
                )
                flip_rates_real["t2i_img"].append(
                    (pred_t2i_clean[real_mask] != pred_t2i_img_adv[real_mask])
                    .float()
                    .mean()
                    .item()
                )
                flip_rates_real["t2i_txt"].append(
                    (pred_t2i_clean[real_mask] != pred_t2i_txt_adv[real_mask])
                    .float()
                    .mean()
                    .item()
                )
            else:
                for key in flip_rates_real.keys():
                    flip_rates_real[key].append(0.0)

            if fake_mask.any():
                flip_rates_fake["img"].append(
                    (pred_img_clean[fake_mask] != pred_img_adv[fake_mask]).float().mean().item()
                )
                flip_rates_fake["i2t_img"].append(
                    (pred_i2t_clean[fake_mask] != pred_i2t_img_adv[fake_mask])
                    .float()
                    .mean()
                    .item()
                )
                flip_rates_fake["i2t_txt"].append(
                    (pred_i2t_clean[fake_mask] != pred_i2t_txt_adv[fake_mask])
                    .float()
                    .mean()
                    .item()
                )
                flip_rates_fake["t2i_img"].append(
                    (pred_t2i_clean[fake_mask] != pred_t2i_img_adv[fake_mask])
                    .float()
                    .mean()
                    .item()
                )
                flip_rates_fake["t2i_txt"].append(
                    (pred_t2i_clean[fake_mask] != pred_t2i_txt_adv[fake_mask])
                    .float()
                    .mean()
                    .item()
                )
            else:
                for key in flip_rates_fake.keys():
                    flip_rates_fake[key].append(0.0)

    avg_loss = total_loss / max(1, total_samples)

    def _avg(lst):
        return sum(lst) / max(1, len(lst))

    avg_flip_img = _avg(flip_rates["img"])
    avg_flip_i2t_img = _avg(flip_rates["i2t_img"])
    avg_flip_i2t_txt = _avg(flip_rates["i2t_txt"])
    avg_flip_t2i_img = _avg(flip_rates["t2i_img"])
    avg_flip_t2i_txt = _avg(flip_rates["t2i_txt"])

    avg_flip_overall = (
        avg_flip_img + avg_flip_i2t_img + avg_flip_i2t_txt + avg_flip_t2i_img + avg_flip_t2i_txt
    ) / 5.0

    avg_flip_img_real = _avg(flip_rates_real["img"])
    avg_flip_img_fake = _avg(flip_rates_fake["img"])
    avg_flip_i2t_img_real = _avg(flip_rates_real["i2t_img"])
    avg_flip_i2t_img_fake = _avg(flip_rates_fake["i2t_img"])
    avg_flip_i2t_txt_real = _avg(flip_rates_real["i2t_txt"])
    avg_flip_i2t_txt_fake = _avg(flip_rates_fake["i2t_txt"])
    avg_flip_t2i_img_real = _avg(flip_rates_real["t2i_img"])
    avg_flip_t2i_img_fake = _avg(flip_rates_fake["t2i_img"])
    avg_flip_t2i_txt_real = _avg(flip_rates_real["t2i_txt"])
    avg_flip_t2i_txt_fake = _avg(flip_rates_fake["t2i_txt"])

    flip_imbalance_ratio = max(
        avg_flip_img_real / max(avg_flip_img_fake, 1e-8),
        avg_flip_img_fake / max(avg_flip_img_real, 1e-8),
    )

    if flip_imbalance_ratio > 5.0:
        print(
            f"⚠ WARNING: Flipping pathology detected! Real/Fake flip ratio: {flip_imbalance_ratio:.1f}"
        )
        print(f"   Real flips: {avg_flip_img_real:.3f}, Fake flips: {avg_flip_img_fake:.3f}")

    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")
    print(f"[Flip Rates] Overall: {avg_flip_overall:.3f}")
    print(
        f"   img: {avg_flip_img:.3f} "
        f"(R: {avg_flip_img_real:.3f}, F: {avg_flip_img_fake:.3f})"
    )
    print(
        f"   i2t_img: {avg_flip_i2t_img:.3f} "
        f"(R: {avg_flip_i2t_img_real:.3f}, F: {avg_flip_i2t_img_fake:.3f})"
    )
    print(
        f"   i2t_txt: {avg_flip_i2t_txt:.3f} "
        f"(R: {avg_flip_i2t_txt_real:.3f}, F: {avg_flip_i2t_txt_fake:.3f})"
    )
    print(
        f"   t2i_img: {avg_flip_t2i_img:.3f} "
        f"(R: {avg_flip_t2i_img_real:.3f}, F: {avg_flip_t2i_img_fake:.3f})"
    )
    print(
        f"   t2i_txt: {avg_flip_t2i_txt:.3f} "
        f"(R: {avg_flip_t2i_txt_real:.3f}, F: {avg_flip_t2i_txt_fake:.3f})"
    )

    return {
        "loss": avg_loss,
        "acc_img": correct_img / max(1, total_samples),
        "acc_i2t": correct_i2t / max(1, total_samples),
        "acc_t2i": correct_t2i / max(1, total_samples),
        "flip_rate_img": avg_flip_img,
        "flip_rate_i2t_img": avg_flip_i2t_img,
        "flip_rate_i2t_txt": avg_flip_i2t_txt,
        "flip_rate_t2i_img": avg_flip_t2i_img,
        "flip_rate_t2i_txt": avg_flip_t2i_txt,
        "flip_rate_overall": avg_flip_overall,
        "flip_rate_img_real": avg_flip_img_real,
        "flip_rate_img_fake": avg_flip_img_fake,
        "flip_rate_i2t_img_real": avg_flip_i2t_img_real,
        "flip_rate_i2t_img_fake": avg_flip_i2t_img_fake,
        "flip_rate_i2t_txt_real": avg_flip_i2t_txt_real,
        "flip_rate_i2t_txt_fake": avg_flip_i2t_txt_fake,
        "flip_rate_t2i_img_real": avg_flip_t2i_img_real,
        "flip_rate_t2i_img_fake": avg_flip_t2i_img_fake,
        "flip_rate_t2i_txt_real": avg_flip_t2i_txt_real,
        "flip_rate_t2i_txt_fake": avg_flip_t2i_txt_fake,
        "flip_imbalance_ratio": flip_imbalance_ratio,
    }


def setup_backbone_unfreezing(model: MMHeadsClassifier, epoch: int, unfreeze_epoch: int = 6):
    backbone = model.mm

    if epoch < unfreeze_epoch:
        for param in backbone.parameters():
            param.requires_grad = False
        print(f"[Freeze] Epoch {epoch}: Backbone fully frozen")
    elif epoch == unfreeze_epoch:
        for name, param in backbone.named_parameters():
            if any(
                key in name
                for key in [
                    "transformer.resblocks.23",
                    "layer4",
                    "blocks.23",
                    "norm",
                ]
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"[Freeze] Epoch {epoch}: Unfrozen last layers of backbone")
    else:
        for param in backbone.parameters():
            param.requires_grad = True
        print(f"[Freeze] Epoch {epoch}: Backbone fully unfrozen")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] Trainable params: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")


@torch.no_grad()
def validate(
    model: MMHeadsClassifier,
    loader: DataLoader,
    device: torch.device,
    templates: List[str],
    epoch: int = 1,
    num_epochs: int = 1,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_labels: List[torch.Tensor] = []
    all_p_img: List[torch.Tensor] = []
    all_p_i2t: List[torch.Tensor] = []
    all_p_t2i: List[torch.Tensor] = []
    correct_img = correct_i2t = correct_t2i = 0

    for batch in tqdm(loader, desc=f"Val {epoch}/{num_epochs}", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        texts_batch = pick_one_caption(batch["texts"])

        logits_img = model(imgs, texts_batch, templates, device, mode="img_only")
        logits_i2t = model(imgs, texts_batch, templates, device, mode="i2t")
        logits_t2i = model(imgs, texts_batch, templates, device, mode="t2i")

        loss = ce(logits_img, labels) + ce(logits_i2t, labels) + ce(logits_t2i, labels)

        B = labels.size(0)
        total_loss += loss.item() * B
        total_samples += B

        probs_img = logits_img.softmax(dim=1)[:, 1]
        probs_i2t = logits_i2t.softmax(dim=1)[:, 1]
        probs_t2i = logits_t2i.softmax(dim=1)[:, 1]

        all_labels.append(labels.cpu())
        all_p_img.append(probs_img.cpu())
        all_p_i2t.append(probs_i2t.cpu())
        all_p_t2i.append(probs_t2i.cpu())

        correct_img += (logits_img.argmax(dim=1) == labels).sum().item()
        correct_i2t += (logits_i2t.argmax(dim=1) == labels).sum().item()
        correct_t2i += (logits_t2i.argmax(dim=1) == labels).sum().item()

    y = torch.cat(all_labels).numpy()
    p_img = torch.cat(all_p_img).numpy()
    p_i2t = torch.cat(all_p_i2t).numpy()
    p_t2i = torch.cat(all_p_t2i).numpy()

    def safe_auc(y_true, y_score):
        try:
            return roc_auc_score(y_true, y_score)
        except Exception:
            return float("nan")

    auc_img = safe_auc(y, p_img)
    auc_i2t = safe_auc(y, p_i2t)
    auc_t2i = safe_auc(y, p_t2i)

    avg_loss = total_loss / max(1, total_samples)
    return {
        "loss": avg_loss,
        "acc_img": correct_img / max(1, total_samples),
        "acc_i2t": correct_i2t / max(1, total_samples),
        "acc_t2i": correct_t2i / max(1, total_samples),
        "auc_img": auc_img,
        "auc_i2t": auc_i2t,
        "auc_t2i": auc_t2i,
    }


def build_mm_backbone(cfg: Dict[str, Any], device: torch.device, init_from: str) -> MMModel:
    model_type = cfg["model"].get("type", "clip")

    mm = MMModel(
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

    if init_from and os.path.isfile(init_from):
        ckpt = torch.load(init_from, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        sd = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in sd.items()
        }
        missing, unexpected = mm.load_state_dict(sd, strict=False)
        print(f"[INIT] Loaded backbone from {init_from}")
        if missing:
            print(f"[INIT] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[INIT] Unexpected keys: {len(unexpected)}")
    else:
        print("[INIT] WARNING: init_from not provided or not found; using randomly init MMModel")

    return mm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--init_from", required=True, help="Path to FULL contrastive checkpoint (.pt)")
    ap.add_argument("--resume_full", default=None, help="Path to FULL classifier checkpoint")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_all(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = cfg.get("clf", {}).get("out_dir", "outputs/checkpoints_mm_heads_vitg14")
    os.makedirs(out_dir, exist_ok=True)

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = False
    if WANDB_AVAILABLE and wandb_cfg.get("enable", False):
        run_name = wandb_cfg.get("run_name", "mm_heads_classifier")
        project = wandb_cfg.get("project", "mm_forensics")
        entity = wandb_cfg.get("entity", None)
        try:
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=cfg,
                mode=wandb_cfg.get("mode", "offline"),
            )
            use_wandb = True
        except Exception as e:
            print(f"[W&B] init failed, disabling logging: {e}", flush=True)
            use_wandb = False

    mm = build_mm_backbone(cfg, device, args.init_from)
    clf_embed_dim = cfg.get("clf", {}).get("embed_dim", 1024)
    clf_model = MMHeadsClassifier(mm, embed_dim=clf_embed_dim, share_pair_head=False).to(device)

    start_epoch = 1
    best_val_auc = -1.0
    resume_full_path = args.resume_full

    data_cfg = cfg["data"]
    frames_root = data_cfg["frames_root"]
    project_root = data_cfg.get("project_root", ".")
    image_size = data_cfg.get("image_size", 224)

    train_ds = FFPPTripletDataset(
        jsonl_path=data_cfg["train_manifest"],
        frames_root=frames_root,
        image_size=image_size,
        augment=True,
        project_root=project_root,
        clip_norm=True,
        return_pair=False,
    )
    val_ds = FFPPTripletDataset(
        jsonl_path=data_cfg["val_manifest"],
        frames_root=frames_root,
        image_size=image_size,
        augment=False,
        project_root=project_root,
        clip_norm=True,
        return_pair=False,
    )

    batch_size = cfg.get("clf", {}).get("batch_size", 64)
    num_workers = cfg.get("clf", {}).get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_mm_classifier,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_mm_classifier,
    )

    lr = cfg.get("clf", {}).get("lr", 1e-4)
    wd = cfg.get("clf", {}).get("weight_decay", 1e-4)
    amp_mode = cfg.get("clf", {}).get("amp", "bf16")
    epochs = cfg.get("clf", {}).get("epochs", 10)

    lr_backbone = cfg.get("clf", {}).get("lr_backbone", lr * 0.2)
    lr_heads = cfg.get("clf", {}).get("lr_heads", lr)

    optimizer_groups = [
        {"params": clf_model.head_img.parameters(), "lr": lr_heads},
        {"params": clf_model.head_i2t.parameters(), "lr": lr_heads},
        {"params": clf_model.head_t2i.parameters(), "lr": lr_heads},
        {"params": clf_model.mm.parameters(), "lr": lr_backbone, "weight_decay": wd * 0.5},
    ]

    optimizer = torch.optim.AdamW(optimizer_groups)
    scaler = GradScaler(enabled=(amp_mode == "fp16"))

    if resume_full_path is not None:
        ckpt_full = torch.load(resume_full_path, map_location="cpu")
        if "model" in ckpt_full:
            clf_model.load_state_dict(ckpt_full["model"])
        else:
            raise KeyError(
                f"[RESUME] Expected 'model' key in {resume_full_path}, found keys: {ckpt_full.keys()}"
            )

        if "optimizer" in ckpt_full:
            try:
                optimizer.load_state_dict(ckpt_full["optimizer"])
                print("[RESUME] Successfully loaded optimizer state")
            except (ValueError, KeyError) as e:
                print(f"[RESUME] Could not load optimizer state: {e}")
                print(
                    "[RESUME] Starting with fresh optimizer (parameter groups changed)"
                )

        if ckpt_full.get("scaler") is not None and amp_mode == "fp16":
            scaler.load_state_dict(ckpt_full["scaler"])

        start_epoch = ckpt_full.get("epoch", 0) + 1
        best_val_auc = ckpt_full.get("best_val_auc", -1.0)
        print(
            f"[RESUME] Loaded FULL classifier from {resume_full_path}, "
            f"resuming at epoch {start_epoch} (best_val_auc={best_val_auc:.3f})",
            flush=True,
        )
    else:
        print("[INIT] Starting classifier training from scratch (no FULL resume).", flush=True)

    templates = cfg["model"].get("clip_prompt_templates_eval", ["{caption}"])

    adv_cfg = cfg.get("clf_adv", {})
    adv_enable = bool(adv_cfg.get("enable", True))
    adv_eps = float(adv_cfg.get("eps", 1.6e-2))
    adv_step = float(adv_cfg.get("step_size", 4.0e-3))
    adv_steps = int(adv_cfg.get("steps", 3))
    text_prob = float(adv_cfg.get("text_prob", 0.5))
    text_max_subs = int(adv_cfg.get("text_max_subs", 3))
    text_mismatch_prob = float(adv_cfg.get("text_mismatch_prob", 0.3))

    def apply_emergency_correction(epoch_stats, current_eps, current_step):
        real_flip = epoch_stats.get("flip_rate_img_real", 0.0)
        fake_flip = epoch_stats.get("flip_rate_img_fake", 0.0)

        if real_flip < 1e-8 or fake_flip < 1e-8:
            return current_eps, current_step

        flip_ratio = max(real_flip / fake_flip, fake_flip / real_flip)

        if flip_ratio > 5.0:
            print(
                f"⚠ EMERGENCY: Flipping pathology detected (ratio: {flip_ratio:.1f}x)"
            )

            new_eps = current_eps * 0.7
            new_step = current_step * 0.7

            if real_flip > fake_flip * 2:
                print(
                    f"   REAL class vulnerable ({real_flip:.3f} vs {fake_flip:.3f})"
                )
                print(
                    f"   Reducing attack strength: eps {current_eps:.4f} -> {new_eps:.4f}"
                )
            else:
                print(
                    f"   FAKE class vulnerable ({fake_flip:.3f} vs {real_flip:.3f})"
                )
                print(
                    f"   Reducing attack strength: eps {current_eps:.4f} -> {new_eps:.4f}"
                )

            return new_eps, new_step

        return current_eps, current_step

    print("=== classifier training (frozen robust backbone + 3 heads) ===")
    print(f"[CFG] epochs={epochs} bs={batch_size} lr={lr} wd={wd} amp={amp_mode}")
    print(
        f"[ADV] enable={adv_enable} eps={adv_eps} step={adv_step} "
        f"steps={adv_steps} text_prob={text_prob} "
        f"text_max_subs={text_max_subs} text_mismatch={text_mismatch_prob}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        if adv_enable and adv_cfg.get("unfreeze_backbone", False):
            unfreeze_epoch = adv_cfg.get("unfreeze_epoch", 6)
            setup_backbone_unfreezing(clf_model, epoch, unfreeze_epoch)

        if adv_enable:
            train_stats = train_one_epoch_adv(
                clf_model,
                train_loader,
                device,
                templates,
                optimizer,
                scaler,
                amp_mode=amp_mode,
                epoch=epoch,
                num_epochs=epochs,
                adv_eps=adv_eps,
                adv_step_size=adv_step,
                adv_steps=adv_steps,
                text_prob=text_prob,
                text_max_subs=text_max_subs,
                text_mismatch_prob=text_mismatch_prob,
                max_adv_batches=adv_cfg.get("max_train_batches"),
            )

            adv_eps, adv_step = apply_emergency_correction(
                train_stats, adv_eps, adv_step
            )
        else:
            train_stats = train_one_epoch(
                clf_model,
                train_loader,
                device,
                templates,
                optimizer,
                scaler,
                amp_mode=amp_mode,
                epoch=epoch,
                num_epochs=epochs,
            )

        val_stats = validate(
            clf_model, val_loader, device, templates, epoch=epoch, num_epochs=epochs
        )

        print(
            f"\n[Epoch {epoch}] train_loss={train_stats['loss']:.4f} "
            f"train_acc_img={train_stats['acc_img']:.3f} "
            f"train_acc_i2t={train_stats['acc_i2t']:.3f} "
            f"train_acc_t2i={train_stats['acc_t2i']:.3f}",
            flush=True,
        )

        if adv_enable and "flip_rate_overall" in train_stats:
            print(
                f"           train_flip={train_stats['flip_rate_overall']:.3f}",
                flush=True,
            )

        print(
            f"           val_clean: acc=({val_stats['acc_img']:.3f},"
            f"{val_stats['acc_i2t']:.3f},{val_stats['acc_t2i']:.3f}) "
            f"auc=({val_stats['auc_img']:.3f},{val_stats['auc_i2t']:.3f},"
            f"{val_stats['auc_t2i']:.3f})",
            flush=True,
        )

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "train/acc_img": train_stats["acc_img"],
                "train/acc_i2t": train_stats["acc_i2t"],
                "train/acc_t2i": train_stats["acc_t2i"],
                "val/loss": val_stats["loss"],
                "val/acc_img": val_stats["acc_img"],
                "val/acc_i2t": val_stats["acc_i2t"],
                "val/acc_t2i": val_stats["acc_t2i"],
                "val/auc_img": val_stats["auc_img"],
                "val/auc_i2t": val_stats["auc_i2t"],
                "val/auc_t2i": val_stats["auc_t2i"],
            }

            if adv_enable:
                log_dict.update(
                    {
                        "train/flip_rate_img": train_stats.get("flip_rate_img", 0),
                        "train/flip_rate_i2t_img": train_stats.get(
                            "flip_rate_i2t_img", 0
                        ),
                        "train/flip_rate_i2t_txt": train_stats.get(
                            "flip_rate_i2t_txt", 0
                        ),
                        "train/flip_rate_t2i_img": train_stats.get(
                            "flip_rate_t2i_img", 0
                        ),
                        "train/flip_rate_t2i_txt": train_stats.get(
                            "flip_rate_t2i_txt", 0
                        ),
                        "train/flip_rate_overall": train_stats.get(
                            "flip_rate_overall", 0
                        ),
                        "train/flip_rate_img_real": train_stats.get(
                            "flip_rate_img_real", 0
                        ),
                        "train/flip_rate_img_fake": train_stats.get(
                            "flip_rate_img_fake", 0
                        ),
                        "train/flip_imbalance_ratio": train_stats.get(
                            "flip_imbalance_ratio", 1.0
                        ),
                    }
                )

            wandb.log(log_dict, step=epoch)

        curr_auc = (
            val_stats["auc_img"] if val_stats["auc_img"] == val_stats["auc_img"] else -1.0
        )
        improved = curr_auc > best_val_auc
        if improved:
            best_val_auc = curr_auc

            best_full_path = os.path.join(out_dir, "mm_heads_BEST_FULL.pt")
            ckpt_best_full = {
                "epoch": epoch,
                "cfg": cfg,
                "model": clf_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if amp_mode == "fp16" else None,
                "best_val_auc": best_val_auc,
            }
            torch.save(ckpt_best_full, best_full_path)

            thin_path = os.path.join(out_dir, "mm_heads_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "cfg": cfg,
                    "classifier_state_dict": clf_model.state_dict(),
                    "best_val_auc": best_val_auc,
                },
                thin_path,
            )
            print(
                f"[CKPT] Saved BEST_FULL to {best_full_path} and thin best to {thin_path} "
                f"(val_auc_img={best_val_auc:.3f})",
                flush=True,
            )

        last_full_path = os.path.join(out_dir, "mm_heads_LAST_FULL.pt")
        ckpt_last_full = {
            "epoch": epoch,
            "cfg": cfg,
            "model": clf_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if amp_mode == "fp16" else None,
            "best_val_auc": best_val_auc,
        }
        torch.save(ckpt_last_full, last_full_path)

    if use_wandb:
        wandb.finish()

    print("\n[REPORT] Training completed. For adversarial evaluation, use evaluate_classifier.py.")


if __name__ == "__main__":
    main()

