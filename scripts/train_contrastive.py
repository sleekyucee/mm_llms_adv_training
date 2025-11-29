# scripts/train_contrastive.py
import os
import sys
import math
import argparse
import random
import signal
import warnings
import gc
import functools
from datetime import datetime
from contextlib import contextmanager

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.utils.checkpoint as _chk

from utils_mm import load_config, seed_all
from logger import Logger
from dataset_mm import FFPPTripletDataset
from model_mm import MMModel
from losses_mm import (
    clip_like_loss,
    text_only_contrastive_loss,
    label_aware_clip_xmodal,
    paraphrase_consensus_loss,
    label_aware_paraphrase_consensus,
    label_aware_i2i,
    label_aware_t2t_core,
    sbi_cross_label_loss,
    image_consistency_loss,
    temperature_reg,
    cross_modal_trades,
    flip_aware_margin,
    trades_t2t,
    trades_i2i,
    mass_losses_light,
    robust_alignment_loss,
    forgery_proto_loss,
    sim_matrix,
    sim_logits,
)

_chk.checkpoint = functools.partial(_chk.checkpoint, use_reentrant=False)

def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def collate_mm(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    images2_list = [b.get("image2", None) for b in batch]
    images2 = torch.stack(images2_list, dim=0) if all(isinstance(x, torch.Tensor) for x in images2_list) else None
    blend_list = [b.get("blend_image", None) for b in batch]
    blend = torch.stack(blend_list, dim=0) if all(isinstance(x, torch.Tensor) for x in blend_list) else None
    texts = [b["texts"] for b in batch]
    labels = torch.tensor([b.get("label", 0) for b in batch], dtype=torch.long)
    source_type = [b.get("source_type", "") for b in batch]
    return {
        "image": images,
        "image2": images2,
        "blend_image": blend,
        "texts": texts,
        "label": labels,
        "source_type": source_type,
    }

def build_text_batch_for_it_balanced(text_groups, step=0):
    picked = []
    for ts in text_groups:
        if not ts:
            picked.append("")
        else:
            picked.append(ts[(step % len(ts))])
    return picked

def encode_with_prompt_ensemble(mm, texts, templates, device):
    outs = []
    for t in texts:
        prompts = [tpl.format(caption=t) for tpl in templates]
        z = mm.encode_text(prompts, style_code=None, device=device)
        z = F.normalize(z.mean(dim=0, keepdim=True), dim=-1)
        outs.append(z)
    return torch.cat(outs, dim=0)

def encode_variants_with_prompts(mm, texts_list, templates, device):
    all_out = []
    for ts in texts_list:
        if not ts:
            all_out.append(None)
            continue
        embs = []
        for t in ts:
            prompts = [tpl.format(caption=t) for tpl in templates]
            z = mm.encode_text(prompts, style_code=None, device=device)
            z = F.normalize(z.mean(dim=0, keepdim=True), dim=-1)
            embs.append(z.squeeze(0))
        all_out.append(torch.stack(embs, dim=0))
    return all_out

def make_two_text_views(variant_embs):
    A, B = [], []
    ref = None
    for emb in variant_embs:
        if emb is not None and emb.numel() > 0:
            ref = emb
            break
    if ref is None:
        return None, None
    D, device = ref.size(-1), ref.device
    zero = torch.zeros(1, D, device=device)
    for emb in variant_embs:
        if emb is None or emb.size(0) == 0:
            A.append(zero); B.append(zero); continue
        idx_even = torch.arange(0, emb.size(0), 2, device=device)
        idx_odd = torch.arange(1, emb.size(0), 2, device=device)
        a = emb[idx_even].mean(dim=0, keepdim=True) if idx_even.numel() > 0 else emb[:1]
        b = emb[idx_odd].mean(dim=0, keepdim=True) if idx_odd.numel() > 0 else emb[-1:].contiguous()
        A.append(F.normalize(a, dim=-1)); B.append(F.normalize(b, dim=-1))
    return torch.cat(A, dim=0), torch.cat(B, dim=0)

def cosine_sim(a: torch.Tensor, b: torch.Tensor):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return a @ b.T

def recall_at_k(sim, k=1):
    topk = sim.topk(k, dim=1).indices
    correct = torch.arange(sim.size(0), device=sim.device)
    hits = (topk == correct[:, None]).any(dim=1).float()
    return hits.mean().item()

def median_rank(sim):
    correct = torch.arange(sim.size(0), device=sim.device)
    ranks = torch.argsort(sim, dim=1, descending=True)
    pos = (ranks == correct[:, None]).nonzero(as_tuple=False)[:, 1]
    return pos.median().item() + 1

def mean_reciprocal_rank(sim):
    correct = torch.arange(sim.size(0), device=sim.device)
    ranks = torch.argsort(sim, dim=1, descending=True)
    pos = (ranks == correct[:, None]).nonzero(as_tuple=False)[:, 1].float() + 1.0
    return (1.0 / pos).mean().item()

def ndcg_at_k(sim, k=5):
    correct = torch.arange(sim.size(0), device=sim.device)
    ranks = torch.argsort(sim, dim=1, descending=True)
    pos = (ranks == correct[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    import math as _m
    vals = [1.0/_m.log2(float(p.item())+1.0) if p.item() <= k else 0.0 for p in pos]
    return float(sum(vals)/len(vals)) if len(vals)>0 else 0.0

def auc_cmc(sim):
    B = sim.size(0)
    correct = torch.arange(B, device=sim.device)
    ranks = torch.argsort(sim, dim=1, descending=True)
    pos = (ranks == correct[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    cmc = torch.zeros(B, device=sim.device)
    for r in range(1, B+1):
        cmc[r-1] = (pos <= r).float().mean()
    return cmc.mean().item()

def pairwise_auroc(sim, sample_neg_frac=0.05, rng=None):
    if rng is None: rng = random.Random(0)
    B = sim.size(0)
    pos = sim.diag().detach().cpu().tolist()
    neg = []
    total_candidates = B*B - B
    sample = max(B, int(sample_neg_frac * total_candidates))
    for i in range(B):
        for j in range(B):
            if i == j: continue
            if rng.random() < float(sample) / float(total_candidates):
                neg.append(sim[i, j].item())
    if len(neg) == 0 or len(pos) == 0:
        return 0.5
    combined = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    combined.sort(key=lambda t: t[0])
    rank = 1
    sum_pos_ranks = 0.0
    i = 0
    N = len(combined)
    while i < N:
        j = i
        while j+1 < N and combined[j+1][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * (rank + j + 1)
        for k in range(i, j+1):
            if combined[k][1] == 1:
                sum_pos_ranks += avg_rank
        rank = j + 2
        i = j + 1
    n_pos = len(pos); n_neg = len(neg)
    auc = (sum_pos_ranks - n_pos*(n_pos+1)/2.0) / (n_pos * n_neg + 1e-12)
    return float(max(0.0, min(1.0, auc)))

def r_at_1_from_sim(sim): return recall_at_k(sim, k=1)
def r_at_5_from_sim(sim): return recall_at_k(sim, k=5)

def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr + 0.5 * (1.0 - min_lr) * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def clamp_logit_scale_if_present(model, lo: float, hi: float):
    try:
        m = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(m, "logit_scale"):
            with torch.no_grad():
                m.logit_scale.clamp_(math.log(lo), math.log(hi))
    except Exception:
        pass

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].data)

def save_full_ckpt(path, *, model, optimizer, scheduler, scaler, ema,
                   cfg, epoch, global_step, best_val, best_rob, bad, bad_rob):
    _ensure_dir(os.path.dirname(path))
    m = model.module if isinstance(model, nn.DataParallel) else model
    state = {
        "ts": _now(), "cfg": cfg,
        "epoch": int(epoch), "global_step": int(global_step),
        "best_val": float(best_val) if best_val is not None else None,
        "best_rob": float(best_rob) if best_rob is not None else None,
        "bad": int(bad), "bad_rob": int(bad_rob),
        "model": m.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema_shadow": getattr(ema, "shadow", None),
    }
    torch.save(state, path)

def make_param_groups(mm: nn.Module, lr_backbone: float, lr_heads: float, weight_decay: float):
    vis, txt, other = [], [], []
    for name, p in mm.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("clip.visual.") or name.startswith("image."):
            vis.append(p)
        elif name.startswith(("clip.transformer.", "clip.token_embedding", "clip.positional_embedding", "clip.ln_final", "clip.text_projection")):
            txt.append(p)
        else:
            other.append(p)
    groups = []
    if vis: groups.append({"params": vis, "lr": lr_backbone, "weight_decay": weight_decay})
    if txt: groups.append({"params": txt, "lr": lr_backbone, "weight_decay": weight_decay})
    if other: groups.append({"params": other, "lr": lr_heads, "weight_decay": weight_decay})
    return groups

def _strip_module_prefix(sd: dict) -> dict:
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def maybe_init_from(model, path):
    if not path or not os.path.isfile(path):
        print("[INIT] training from scratch", flush=True); return
    print(f"[INIT] loading weights from {path}", flush=True)
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[INIT][WARN] torch.load failed: {e} (training from scratch)", flush=True); return
    sd_raw = ckpt.get("model", ckpt)
    sd = _strip_module_prefix(sd_raw)
    target = model.module if isinstance(model, nn.DataParallel) else model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(f"[INIT] loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)

@torch.no_grad()
def clamp_like_clip(x):
    return torch.clamp(x, -10.0, 10.0)

def _cosine_to_target(z, t):
    z = F.normalize(z, dim=-1); t = F.normalize(t, dim=-1)
    return (z * t).sum(dim=1).mean()

def pgd_attack_images(
    model, imgs, texts_or_none, cfg, device, tpl,
    *, target_emb: torch.Tensor = None,
    objective: str = "i2t",
    eval_mode: bool = False, restart_id: int = 0
):
    m = model.module if isinstance(model, nn.DataParallel) else model
    was_training = m.training

    img_cfg = cfg["eval_adv"]["pgd"] if (eval_mode and ("eval_adv" in cfg and "pgd" in cfg["eval_adv"])) else cfg["adv"]["img"]
    steps = int(img_cfg.get("steps", 12))
    alpha = float(img_cfg.get("step_size", 2e-3))
    eps = float(img_cfg.get("eps", 1.2e-2))
    use_linf = (img_cfg.get("norm", "linf").lower() == "linf")
    clamp_flag = bool(img_cfg.get("clamp", True))
    random_start = bool(img_cfg.get("random_start", True))
    momentum = float(img_cfg.get("momentum", 0.0))
    use_eot = bool(img_cfg.get("eot", False))
    eot_iters = int(img_cfg.get("eot_iters", 1)) if use_eot else 1

    imgs = imgs.detach()
    x_adv = imgs.clone().detach()

    if objective == "i2t":
        with torch.no_grad():
            txt_z = encode_with_prompt_ensemble(m, texts_or_none, tpl, device)
        if txt_z.size(0) != imgs.size(0):
            raise ValueError("i2t attack mismatch between texts and images")
    else:
        txt_z = None

    t_fix = None
    if objective in ("toward_emb", "away_from_emb"):
        assert target_emb is not None and target_emb.size(0) == imgs.size(0), "target_emb must be (B,D)"
        t_fix = target_emb.detach()

    if random_start:
        if use_linf:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        else:
            noise = torch.randn_like(x_adv)
            noise = noise / (noise.view(noise.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12)
            x_adv = x_adv + eps * noise
        x_adv = x_adv.detach()

    velocity = torch.zeros_like(x_adv) if momentum > 0 else None

    prev_requires = [p.requires_grad for p in m.parameters()]
    for p in m.parameters():
        p.requires_grad_(False)

    try:
        with temporarily_disable_checkpointing(m):
            m.eval()
            for _ in range(steps):
                x_adv = x_adv.detach().requires_grad_(True)
                with torch.enable_grad(), autocast(enabled=False):
                    total_loss = 0.0
                    for _e in range(eot_iters):
                        x_e = x_adv
                        if use_eot and (torch.rand(1).item() < 0.5):
                            h, w = x_e.shape[-2:]
                            scale = random.choice([0.9, 1.0, 1.1])
                            nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
                            x_e = F.interpolate(x_e, size=(nh, nw), mode="bilinear", align_corners=False)
                            x_e = F.interpolate(x_e, size=(h, w), mode="bilinear", align_corners=False)
                        img_z = m.encode_image(x_e.float())
                        if objective == "i2t":
                            loss_e = -clip_like_loss(img_z, txt_z, m.get_logit_scale())
                        elif objective == "toward_emb":
                            loss_e = _cosine_to_target(img_z, t_fix) * m.get_logit_scale()
                        elif objective == "away_from_emb":
                            loss_e = -_cosine_to_target(img_z, t_fix) * m.get_logit_scale()
                        else:
                            raise ValueError(f"Unknown objective: {objective}")
                        total_loss = total_loss + loss_e
                    loss = total_loss / float(eot_iters)

                if x_adv.grad is not None:
                    x_adv.grad.zero_()
                for p in m.parameters(): p.grad = None
                loss.backward()
                grad = x_adv.grad.detach() if x_adv.grad is not None else torch.zeros_like(x_adv)

                if momentum > 0:
                    if use_linf:
                        grad = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-12)
                    else:
                        grad = grad / (grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-12)
                    velocity = momentum * velocity + grad
                    g_use = velocity
                else:
                    g_use = grad

                if use_linf:
                    x_adv = x_adv + alpha * g_use.sign()
                    delta = (x_adv - imgs).clamp_(-eps, eps)
                else:
                    g_unit = g_use / (g_use.view(g_use.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-12)
                    x_adv = x_adv + alpha * g_unit
                    diff = x_adv - imgs
                    nrm = diff.view(diff.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1)
                    scale = (eps / torch.maximum(nrm, torch.tensor(1e-12, device=device))).clamp(max=1.0)
                    delta = diff * scale
                x_adv = (imgs + delta).detach()
    finally:
        for p, flag in zip(m.parameters(), prev_requires):
            p.requires_grad_(flag)
        m.train(was_training)

    if clamp_flag:
        x_adv = clamp_like_clip(x_adv)
    return x_adv.detach()

@contextmanager
def temporarily_disable_checkpointing(model: nn.Module):
    m = model.module if isinstance(model, nn.DataParallel) else model
    saved_flags = {}
    for attr in ["grad_checkpoint", "gradient_checkpointing", "use_checkpoint", "checkpointing"]:
        if hasattr(m, attr):
            try:
                saved_flags[attr] = getattr(m, attr)
                setattr(m, attr, False)
            except Exception:
                pass
    try:
        yield
    finally:
        for k, v in saved_flags.items():
            try: setattr(m, k, v)
            except Exception: pass

def _strong_text_perturb(batch_texts, prob=0.5, max_subs=3, mismatch_prob=0.3, rng=None):
    if rng is None: rng = random
    B = len(batch_texts)
    out = []
    roll = list(range(1, B)) + [0]
    for i, t in enumerate(batch_texts):
        s = t
        if s and rng.random() < prob:
            toks = s.split()
            if len(toks) >= 1:
                j = rng.randrange(0, len(toks))
                toks[j] = toks[j] + " not" if rng.random() < 0.25 else toks[j]
            if rng.random() < mismatch_prob:
                s = batch_texts[roll[i]]
            else:
                s = " ".join(toks)
        out.append(s)
    return out

def should_run_eval_adv(cfg, epoch_rel: int) -> bool:
    eg = cfg.get("eval_adv", {}).get("gating", {})
    if not eg: return True
    start = int(eg.get("start_epoch", 1)); k = int(eg.get("every_k_epochs", 1))
    if epoch_rel < start: return False
    return ((epoch_rel - start) % max(1, k)) == 0

def _rolled_indices(n: int, device=None):
    if n <= 1: return torch.zeros(1, dtype=torch.long, device=device)
    return torch.cat([torch.arange(1, n, device=device), torch.tensor([0], device=device)], dim=0)

def eval_adv_all(mm, loader, device, tpl_eval, adv_cfg, text_cfg, cfg, *, total=None, desc="AdvEval"):
    R = {k: 0.0 for k in [
        "R@1_i2t_adv","R@5_i2t_adv","MR_i2t_adv","MRR_i2t_adv","NDCG@5_i2t_adv","AUCMC_i2t_adv","AUROC_i2t_adv",
        "R@1_t2i_adv","R@5_t2i_adv","MR_t2i_adv","MRR_t2i_adv","NDCG@5_t2i_adv","AUCMC_t2i_adv","AUROC_t2i_adv",
        "R@1_i2i_adv","R@5_i2i_adv","MR_i2i_adv","MRR_i2i_adv","NDCG@5_i2i_adv","AUCMC_i2i_adv","AUROC_i2i_adv",
        "R@1_t2t_adv","R@5_t2t_adv","MR_t2t_adv","MRR_t2t_adv","NDCG@5_t2t_adv","AUCMC_t2t_adv","AUROC_t2t_adv",
    ]}
    vc = 0

    model = mm.module if isinstance(mm, nn.DataParallel) else mm
    e_steps = int(adv_cfg.get("steps", 12))
    e_alpha = float(adv_cfg.get("step_size", 2e-3))
    e_eps = float(adv_cfg.get("eps", 1.2e-2))
    e_norm = adv_cfg.get("norm", "linf")

    it = tqdm(
        loader,
        total=total,
        desc=desc,
        position=0,
        leave=True,
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

        texts_list = batch["texts"]
        picked_clean = build_text_batch_for_it_balanced(texts_list, step=0)

        tprob = float(text_cfg.get("prob", 0.5)) if bool(text_cfg.get("enable", True)) else 0.0
        tsubs = int(text_cfg.get("max_subs", 3))
        picked_adv_a = _strong_text_perturb(picked_clean, prob=tprob, max_subs=tsubs, mismatch_prob=0.30)
        picked_adv_b = _strong_text_perturb(picked_clean, prob=tprob, max_subs=tsubs, mismatch_prob=0.30)

        imgs_adv = pgd_attack_images(
            model, imgs, picked_adv_a,
            {"adv": {"img": {"steps": e_steps, "step_size": e_alpha, "eps": e_eps, "norm": e_norm, "clamp": True}}},
            device, tpl_eval, objective="i2t"
        )
        with torch.no_grad():
            img_z_a = model.encode_image(imgs_adv)
            txt_z_a = encode_with_prompt_ensemble(model, picked_adv_a, tpl_eval, device)
        sim_i2t = cosine_sim(img_z_a, txt_z_a)
        R["R@1_i2t_adv"] += r_at_1_from_sim(sim_i2t); R["R@5_i2t_adv"] += r_at_5_from_sim(sim_i2t); R["MR_i2t_adv"] += median_rank(sim_i2t)
        R["MRR_i2t_adv"] += mean_reciprocal_rank(sim_i2t); R["NDCG@5_i2t_adv"] += ndcg_at_k(sim_i2t,5); R["AUCMC_i2t_adv"] += auc_cmc(sim_i2t); R["AUROC_i2t_adv"] += pairwise_auroc(sim_i2t)

        sim_t2i = sim_i2t.T
        R["R@1_t2i_adv"] += r_at_1_from_sim(sim_t2i); R["R@5_t2i_adv"] += r_at_5_from_sim(sim_t2i); R["MR_t2i_adv"] += median_rank(sim_t2i)
        R["MRR_t2i_adv"] += mean_reciprocal_rank(sim_t2i); R["NDCG@5_t2i_adv"] += ndcg_at_k(sim_t2i,5); R["AUCMC_t2i_adv"] += auc_cmc(sim_t2i); R["AUROC_t2i_adv"] += pairwise_auroc(sim_t2i)

        with torch.no_grad():
            if isinstance(imgs2, torch.Tensor):
                img_z_b_clean = model.encode_image(imgs2)
            else:
                img_z_b_clean = model.encode_image(imgs)
            roll = _rolled_indices(img_z_b_clean.size(0), device=img_z_b_clean.device)
            wrong_targets = img_z_b_clean[roll].detach()

        imgs_adv_i2i = pgd_attack_images(
            model, imgs, None,
            {"adv": {"img": {"steps": e_steps, "step_size": e_alpha, "eps": e_eps, "norm": e_norm, "clamp": True}}},
            device, tpl_eval, target_emb=wrong_targets, objective="toward_emb"
        )
        with torch.no_grad():
            img_z_a_targ = model.encode_image(imgs_adv_i2i)
            sim_i2i = cosine_sim(img_z_a_targ, img_z_b_clean)
        R["R@1_i2i_adv"] += r_at_1_from_sim(sim_i2i); R["R@5_i2i_adv"] += r_at_5_from_sim(sim_i2i); R["MR_i2i_adv"] += median_rank(sim_i2i)
        R["MRR_i2i_adv"] += mean_reciprocal_rank(sim_i2i); R["NDCG@5_i2i_adv"] += ndcg_at_k(sim_i2i,5); R["AUCMC_i2i_adv"] += auc_cmc(sim_i2i); R["AUROC_i2i_adv"] += pairwise_auroc(sim_i2i)

        with torch.no_grad():
            txtA = encode_with_prompt_ensemble(model, picked_adv_a, tpl_eval, device)
            txtB = encode_with_prompt_ensemble(model, picked_adv_b, tpl_eval, device)
            sim_t2t = cosine_sim(txtA, txtB)
        R["R@1_t2t_adv"] += r_at_1_from_sim(sim_t2t); R["R@5_t2t_adv"] += r_at_5_from_sim(sim_t2t); R["MR_t2t_adv"] += median_rank(sim_t2t)
        R["MRR_t2t_adv"] += mean_reciprocal_rank(sim_t2t); R["NDCG@5_t2t_adv"] += ndcg_at_k(sim_t2t,5); R["AUCMC_t2t_adv"] += auc_cmc(sim_t2t); R["AUROC_t2t_adv"] += pairwise_auroc(sim_t2t)

        vc += 1

    for k in list(R.keys()):
        R[k] = R[k] / max(vc, 1)

    return R

class AWP:
    def __init__(self, model, optimizer, adv_lr=1e-3, adv_eps=1e-3, param_filter=lambda n,p: p.requires_grad and p.dim() > 1):
        self.model = model.module if isinstance(model, nn.DataParallel) else model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.param_filter = param_filter
    @torch.no_grad()
    def perturb(self):
        self.backup.clear()
        for n, p in self.model.named_parameters():
            if not self.param_filter(n, p): continue
            if p.grad is None: continue
            grad = p.grad
            if grad.norm() == 0: continue
            self.backup[n] = p.data.clone()
            eps = self.adv_eps * (p.data.abs().mean() + 1e-12)
            step = self.adv_lr * grad / (grad.norm() + 1e-12)
            p.add_(step).clamp_(self.backup[n] - eps, self.backup[n] + eps)
    @torch.no_grad()
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup.clear()

def quick_adv_i2t_images(mm, imgs, txt_z, cfg, *, amp_enabled, amp_dtype):
    trades_quick_eps = float(cfg.get("robust", {}).get("trades_eps", 4e-3))
    trades_quick_alpha = float(cfg.get("robust", {}).get("trades_alpha", trades_quick_eps))
    imgs_q = imgs.detach().clone().requires_grad_(True)
    txt_z_det = txt_z.detach()
    req_backup = [p.requires_grad for p in mm.parameters()]
    for p in mm.parameters(): p.requires_grad_(False)
    with temporarily_disable_checkpointing(mm):
        with torch.enable_grad(), autocast(enabled=amp_enabled, dtype=amp_dtype):
            img_z_q = mm.encode_image(imgs_q)
            logits = mm.get_logit_scale() * (F.normalize(img_z_q, dim=-1) @ F.normalize(txt_z_det, dim=-1).T)
            loss_q = -torch.diag(logits).mean()
        if imgs_q.grad is not None: imgs_q.grad.zero_()
        loss_q.backward()
        grad = imgs_q.grad.detach()
    for p, flag in zip(mm.parameters(), req_backup):
        p.requires_grad_(flag)
    if grad is None: return imgs.detach()
    imgs_step = (imgs.detach() + trades_quick_alpha * grad.sign())
    delta = (imgs_step - imgs).clamp(-trades_quick_eps, trades_quick_eps)
    return clamp_like_clip((imgs + delta).detach())

def quick_adv_i2i_images(mm, imgs, img_pos, cfg, *, amp_enabled, amp_dtype, objective="away"):
    trades_quick_eps = float(cfg.get("robust", {}).get("trades_eps", 4e-3))
    trades_quick_alpha = float(cfg.get("robust", {}).get("trades_alpha", trades_quick_eps))
    imgs_q = imgs.detach().clone().requires_grad_(True)
    req_backup = [p.requires_grad for p in mm.parameters()]
    for p in mm.parameters(): p.requires_grad_(False)
    with temporarily_disable_checkpointing(mm):
        with torch.enable_grad(), autocast(enabled=amp_enabled, dtype=amp_dtype):
            z_q = mm.encode_image(imgs_q)
            z_pos = mm.encode_image(img_pos).detach()
            sim = (F.normalize(z_q, dim=-1) * F.normalize(z_pos, dim=-1)).sum(dim=1).mean()
            loss_q = sim if objective == "away" else -sim
        if imgs_q.grad is not None: imgs_q.grad.zero_()
        loss_q.backward()
        grad = imgs_q.grad.detach()
    for p, flag in zip(mm.parameters(), req_backup):
        p.requires_grad_(flag)
    if grad is None: return imgs.detach()
    imgs_step = (imgs.detach() + trades_quick_alpha * grad.sign())
    delta = (imgs_step - imgs).clamp(-trades_quick_eps, trades_quick_eps)
    return clamp_like_clip((imgs + delta).detach())

def _piecewise_schedule(epoch_rel: int, schedule_pairs, default_val: float):
    v = float(default_val)
    if not schedule_pairs: return v
    try:
        for e, val in sorted(schedule_pairs, key=lambda x: int(x[0])):
            if int(epoch_rel) >= int(e): v = float(val)
    except Exception: pass
    return v

def get_attack_prob(cfg, epoch_rel: int) -> float:
    img = cfg.get("adv", {}).get("img", {})
    return _piecewise_schedule(epoch_rel, img.get("attack_prob_schedule", []), img.get("attack_prob", 0.5))

def get_train_mixture_prob(cfg, epoch_rel: int) -> float:
    img = cfg.get("adv", {}).get("img", {})
    return _piecewise_schedule(epoch_rel, img.get("train_mixture_schedule", []), img.get("train_mixture", 0.5))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--init_from", default=None, help="Path to a .pt weights or FULL checkpoint to resume")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("=== startup ===", flush=True)
    print(f"[LAUNCH] {_abs(__file__)}", flush=True)

    seed_all(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_path = args.init_from or cfg.get("train", {}).get("init_from", None)
    print(f"[RESUME] init_from={init_path if init_path else 'None (scratch)'}", flush=True)

    model_type = cfg["model"].get("type", "clip")
    clip_norm = (model_type == "clip")

    train_ds = FFPPTripletDataset(
        cfg["data"]["train_manifest"], cfg["data"]["frames_root"],
        image_size=cfg["data"].get("image_size", 224), augment=True,
        clip_norm=clip_norm, return_pair=True
    )
    val_ds = FFPPTripletDataset(
        cfg["data"]["val_manifest"], cfg["data"]["frames_root"],
        image_size=cfg["data"].get("image_size", 224), augment=False,
        clip_norm=clip_norm, return_pair=True
    )

    numw = min(cfg["train"].get("num_workers", 4), 8)
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))

    def _worker_init_fn(worker_id):
        torch.manual_seed(cfg.get("seed", 42) + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=numw,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_mm,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=numw,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_mm,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )

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
        novelty_cfg=cfg.get("novelty", {})
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[GPU] Using {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)

    mm = model.module if isinstance(model, nn.DataParallel) else model

    lr_global = float(cfg["train"].get("lr", 1e-5))
    lr_backbone = float(cfg["train"].get("lr_backbone", lr_global))
    lr_heads = float(cfg["train"].get("lr_heads", lr_global))
    weight_decay = float(cfg["train"].get("weight_decay", 1.5e-4))

    param_groups = make_param_groups(mm, lr_backbone, lr_heads, weight_decay)
    if not param_groups:
        param_groups = [{
            "params": [p for p in mm.parameters() if p.requires_grad],
            "lr": lr_global,
            "weight_decay": weight_decay,
        }]
    optimizer = torch.optim.AdamW(param_groups)

    amp_mode = str(cfg["train"].get("amp", "bf16")).lower()
    amp_enabled = (amp_mode != "off") and torch.cuda.is_available()
    amp_dtype = (
        torch.float16 if amp_mode == "fp16"
        else (torch.bfloat16 if amp_mode == "bf16" else None)
    )
    scaler = GradScaler(enabled=(amp_mode == "fp16" and torch.cuda.is_available()))

    accum_steps = int(cfg["train"].get("accum_steps", 1))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    ema_decay = cfg["train"].get("ema_decay", 0.999)
    ema = ModelEMA(mm, decay=ema_decay)

    run_name = cfg.get("exp", {}).get("name", "mm_run_mass")
    logger = Logger(run_name)
    logger.watch(model)

    arch_tag = str(cfg["model"].get("clip_model_name", "clip")).replace("/", "-")
    ckpt_dir = _abs(cfg["train"]["ckpt_dir"])
    _ensure_dir(ckpt_dir)

    tpl_train = cfg["model"].get("clip_prompt_templates_train", ["{caption}"])
    tpl_eval = cfg["model"].get("clip_prompt_templates_eval", ["{caption}"])

    loss_cfg = cfg.get("loss", {})
    robust_cfg = cfg.get("robust", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.10))
    pos_align_w = float(loss_cfg.get("pos_align", 0.05))
    w_txt_margin = float(loss_cfg.get("w_txt_margin", 0.25))

    w_label_clip_xm = float(loss_cfg.get("w_label_clip_xm", 0.005))
    w_label_para_t2t = float(loss_cfg.get("w_label_para_t2t", 0.01))
    w_sbi_xlbl = float(loss_cfg.get("w_sbi_xlbl", 0.05))

    trades_w = float(robust_cfg.get("trades_w", 0.5))
    trades_i2i_w = float(robust_cfg.get("trades_i2i_w", 0.0))
    trades_T = float(robust_cfg.get("trades_T", 1.0))
    trades_t2t_w = float(robust_cfg.get("trades_t2t_w", 0.25))
    fap_w = float(robust_cfg.get("fap_w", 0.30))
    fap_margin = float(robust_cfg.get("fap_margin", 0.10))

    mass_cfg = robust_cfg.get("mass", {})
    mass_enabled = bool(mass_cfg.get("enable", mass_cfg.get("enabled", False)))
    mass_w_trades = float(mass_cfg.get("w_trades_smooth", 0.02))
    mass_w_cons = float(mass_cfg.get("w_consistency", 0.01))

    awp_cfg = robust_cfg.get("awp", {"enable": False})
    use_awp = bool(awp_cfg.get("enable", False))
    awp = AWP(
        model,
        optimizer,
        adv_lr=float(awp_cfg.get("adv_lr", 1e-3)),
        adv_eps=float(awp_cfg.get("adv_eps", 1e-3)),
    ) if use_awp else None

    best_val = None
    best_rob = None
    bad = 0
    bad_rob = 0
    start_epoch_abs = 1
    global_step = 0
    steps_per_epoch = max(1, len(train_loader))
    loaded_full = False

    if init_path and os.path.isfile(init_path):
        try:
            ckpt = torch.load(init_path, map_location="cpu")
        except Exception as e:
            ckpt = None
            print(f"[INIT][WARN] failed to load: {e}", flush=True)

        if ckpt and isinstance(ckpt, dict) and all(
            k in ckpt for k in ["model", "optimizer", "scheduler", "scaler"]
        ):
            print(f"[INIT] FULL checkpoint detected -> {init_path}", flush=True)
            sd = _strip_module_prefix(ckpt["model"])
            (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(sd, strict=False)

            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])

            if ckpt.get("scaler") is not None and amp_mode == "fp16":
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    print(f"[INIT][WARN] scaler load failed: {e}", flush=True)

            past_epochs = int(ckpt.get("epoch", 0))
            total_epochs_abs = past_epochs + int(cfg["train"].get("epochs", 1))
            total_steps_abs = total_epochs_abs * steps_per_epoch
            warmup_steps_full = int(cfg["train"].get(
                "warmup_steps",
                max(10, int(0.06 * total_steps_abs))
            ))
            min_lr = float(cfg["train"].get("lr_min", 0.0))
            scheduler = make_warmup_cosine_scheduler(
                optimizer, warmup_steps_full, total_steps_abs, min_lr=min_lr
            )
            if ckpt.get("scheduler") is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception as e:
                    print(f"[INIT][WARN] scheduler load failed: {e}", flush=True)

            lr_global = float(cfg["train"].get("lr", 1e-5))
            lr_backbone = float(cfg["train"].get("lr_backbone", lr_global))
            lr_heads = float(cfg["train"].get("lr_heads", lr_global))

            num_groups = len(optimizer.param_groups)
            for gi, g in enumerate(optimizer.param_groups):
                if num_groups <= 1:
                    g["lr"] = lr_global
                elif num_groups == 2:
                    g["lr"] = lr_backbone if gi == 0 else lr_heads
                else:
                    g["lr"] = lr_backbone if gi < (num_groups - 1) else lr_heads

            if scheduler is not None and hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [g["lr"] for g in optimizer.param_groups]

            print("[INIT] LR override after FULL resume:",
                  [g["lr"] for g in optimizer.param_groups],
                  flush=True)

            if ckpt.get("ema_shadow") is not None:
                try:
                    ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
                except Exception as e:
                    print(f"[INIT][WARN] EMA load failed: {e}", flush=True)

            start_epoch_abs = past_epochs + 1
            global_step = int(ckpt.get("global_step", 0))
            best_val = ckpt.get("best_val", None)
            best_rob = ckpt.get("best_rob", None)
            bad = int(ckpt.get("bad", 0) or 0)
            bad_rob = int(ckpt.get("bad_rob", 0) or 0)
            loaded_full = True
            print(f"[INIT] Resume @ epoch={start_epoch_abs}, step={global_step}", flush=True)

    if not loaded_full:
        maybe_init_from(model, init_path)
        total_steps = int(cfg["train"]["epochs"]) * steps_per_epoch
        warmup_steps = int(cfg["train"].get(
            "warmup_steps",
            max(10, int(0.06 * total_steps))
        ))
        min_lr = float(cfg["train"].get("lr_min", 0.0))
        scheduler = make_warmup_cosine_scheduler(
            optimizer, warmup_steps, total_steps, min_lr=min_lr
        )

    run_state = {
        "epoch": start_epoch_abs,
        "global_step": global_step,
        "best_val": best_val,
        "best_rob": best_rob,
        "bad": bad,
        "bad_rob": bad_rob,
    }

    interrupt_path = _abs(os.path.join(ckpt_dir, f"{arch_tag}_interrupt_FULL.pt"))

    def _interrupt_handler(signum, frame):
        try:
            save_full_ckpt(
                interrupt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                cfg=cfg,
                epoch=max(1, run_state["epoch"] - 1),
                global_step=run_state["global_step"],
                best_val=run_state["best_val"],
                best_rob=run_state["best_rob"],
                bad=run_state["bad"],
                bad_rob=run_state["bad_rob"],
            )
        finally:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)

    remaining_epochs = int(cfg["train"]["epochs"])
    end_epoch_abs = start_epoch_abs + remaining_epochs - 1
    enable_early_stop = bool(cfg.get("train", {}).get("enable_early_stop", False))

    for epoch_abs in range(start_epoch_abs, end_epoch_abs + 1):
        epoch_rel = (epoch_abs - start_epoch_abs) + 1
        run_state["epoch"] = epoch_abs

        mm.set_backbone_trainable(
            epoch_rel > int(cfg["train"].get("freeze_img_epochs", 0))
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        nstep = 0

        t_cfg = cfg.get("text", cfg.get("adv", {}).get("text", {}))
        t_en = bool(t_cfg.get("enable", False))
        t_prob = float(t_cfg.get("prob", 0.35))
        t_subs = int(t_cfg.get("max_subs", 3))
        t_mis = float(t_cfg.get("mismatch_prob", 0.30))

        attack_frac = float(cfg.get("adv", {}).get("img", {}).get("attack_frac", 0.5))
        attack_prob = get_attack_prob(cfg, epoch_rel)
        p_i2t = get_train_mixture_prob(cfg, epoch_rel)
        p_away = float(cfg.get("adv", {}).get("img", {}).get("away_frac", 0.0))
        p_toward = max(0.0, 1.0 - (p_i2t + p_away))
        total_mix = max(1e-6, p_i2t + p_away + p_toward)
        p_i2t, p_away, p_toward = (
            p_i2t / total_mix,
            p_away / total_mix,
            p_toward / total_mix,
        )

        train_sum = {
            "total": 0.0,
            "it": 0.0, "tt": 0.0, "ii": 0.0, "ic": 0.0,
            "pos": 0.0, "temp": 0.0,
            "trades": 0.0, "t2t": 0.0, "i2i_trades": 0.0, "fap": 0.0,
            "mass_trades": 0.0, "mass_cons": 0.0, "mass_proto": 0.0,
            "la_clip_xm": 0.0,
            "la_para_t2t": 0.0,
            "la_sbi": 0.0,
            "txt_only": 0.0,
            "la_i2i": 0.0,
            "la_t2t_core": 0.0,
            "robust_align": 0.0,
            "forgery_proto": 0.0,
        }
        train_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"E{epoch_abs}",
            position=0,
            leave=True,
            ncols=60,
            dynamic_ncols=False,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            mininterval=1.0,
            smoothing=0,
        )

        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            imgs2 = batch["image2"]
            imgs2 = imgs2.to(device, non_blocking=True) if isinstance(imgs2, torch.Tensor) else None
            blend = batch["blend_image"]
            blend = blend.to(device, non_blocking=True) if isinstance(blend, torch.Tensor) else None
            texts_list = batch["texts"]

            picked_clean = build_text_batch_for_it_balanced(
                texts_list,
                step=run_state["global_step"],
            )
            picked = (
                _strong_text_perturb(
                    picked_clean,
                    prob=t_prob,
                    max_subs=t_subs,
                    mismatch_prob=t_mis,
                )
                if t_en else picked_clean
            )

            if (
                bool(cfg.get("adv", {}).get("enable", False))
                and ("img" in cfg.get("adv", {}))
                and (torch.rand(1).item() < attack_prob)
            ):
                B = imgs.size(0)
                k = max(1, int(attack_frac * B))
                idx = torch.randperm(B, device=imgs.device)[:k]
                imgs_attack = imgs[idx]

                r = torch.rand(1).item()
                if r < p_i2t:
                    picked_sub = [picked[i.item()] for i in idx]
                    imgs_attacked = pgd_attack_images(
                        model,
                        imgs_attack,
                        picked_sub,
                        cfg,
                        device,
                        tpl_train,
                        objective="i2t",
                        eval_mode=False,
                    )
                elif r < (p_i2t + p_away):
                    with torch.no_grad():
                        true_targets = mm.encode_image(imgs_attack).detach()
                    imgs_attacked = pgd_attack_images(
                        model,
                        imgs_attack,
                        None,
                        cfg,
                        device,
                        tpl_train,
                        target_emb=true_targets,
                        objective="away_from_emb",
                        eval_mode=False,
                    )
                else:
                    with torch.no_grad():
                        gal = imgs2[idx] if isinstance(imgs2, torch.Tensor) else imgs_attack
                        z_gal = mm.encode_image(gal)
                        roll = torch.arange(z_gal.size(0), device=z_gal.device)
                        roll = torch.cat([roll[1:], roll[:1]], dim=0)
                        wrong_targets = z_gal[roll].detach()
                    imgs_attacked = pgd_attack_images(
                        model,
                        imgs_attack,
                        None,
                        cfg,
                        device,
                        tpl_train,
                        target_emb=wrong_targets,
                        objective="toward_emb",
                        eval_mode=False,
                    )

                imgs = imgs.detach()
                imgs[idx] = imgs_attacked

            def compute_total_and_parts():
                w_it = float(loss_cfg.get("w_it", 1.0))
                w_tt = float(loss_cfg.get("w_tt", 0.30))
                w_ii = float(loss_cfg.get("w_ii", 0.25))
                w_text_only = float(loss_cfg.get("w_text_only", 0.0))
                w_label_clip_xm = float(loss_cfg.get("w_label_clip_xm", 0.0))
                w_label_para_t2t = float(loss_cfg.get("w_label_para_t2t", 0.0))
                w_sbi_xlbl = float(loss_cfg.get("w_sbi_xlbl", 0.0))
                w_label_i2i = float(loss_cfg.get("w_label_i2i", 0.0))
                w_label_t2t_core = float(loss_cfg.get("w_label_t2t_core", 0.0))

                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    img_z_raw = mm.encode_image(imgs)
                    txt_z_raw = encode_with_prompt_ensemble(mm, picked, tpl_train, device)

                    refiner = getattr(mm, "refine_embeddings", None)
                    if callable(refiner):
                        img_z, txt_z = refiner(img_z_raw, txt_z_raw, imgs)
                    else:
                        img_z, txt_z = img_z_raw, txt_z_raw

                    if isinstance(blend, torch.Tensor):
                        img_pos_img = blend
                    elif isinstance(imgs2, torch.Tensor):
                        img_pos_img = imgs2
                    else:
                        img_pos_img = imgs
                    img_pos = mm.encode_image(img_pos_img)

                    labels = batch["label"].to(device)

                    loss_it = clip_like_loss(
                        img_z, txt_z, mm.get_logit_scale(),
                        label_smoothing, assume_normalized=True,
                    )

                    if w_text_only > 0.0:
                        loss_txt_only = text_only_contrastive_loss(
                            txt_z=txt_z,
                            temperature=float(loss_cfg.get("text_only_T", 2.0)),
                            assume_normalized=True,
                        )
                    else:
                        loss_txt_only = torch.tensor(0.0, device=device)

                    variant_embs = encode_variants_with_prompts(
                        mm, texts_list, tpl_train, device
                    )
                    loss_tt = paraphrase_consensus_loss(
                        variant_embs, mm.get_logit_scale(),
                        assume_normalized=True,
                    )

                    loss_ii = (
                        clip_like_loss(
                            img_z, img_pos, mm.get_logit_scale(),
                            label_smoothing, assume_normalized=True,
                        )
                        if img_pos is not None else torch.tensor(0.0, device=device)
                    )

                    loss_ic = (
                        image_consistency_loss(
                            img_z, img_pos,
                            weight=loss_cfg.get("img_consistency", 0.1),
                            assume_normalized=True,
                        )
                        if img_pos is not None else torch.tensor(0.0, device=device)
                    )

                    loss_pos = (
                        (
                            1.0 - (
                                F.normalize(img_z, dim=-1)
                                * F.normalize(txt_z, dim=-1)
                            ).sum(dim=1)
                        ).mean() * pos_align_w
                        if pos_align_w > 0 else torch.tensor(0.0, device=device)
                    )

                    loss_temp = temperature_reg(
                        mm.get_logit_scale(),
                        target=float(loss_cfg.get("temp_target", 100.0)),
                        weight=float(loss_cfg.get("temp_reg", 5.0e-4)),
                    )

                    imgs_quick_i2t = quick_adv_i2t_images(
                        mm, imgs, txt_z, cfg,
                        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    )
                    with torch.no_grad():
                        img_z_adv_i2t = mm.encode_image(imgs_quick_i2t)

                    picked_trades = _strong_text_perturb(
                        picked_clean,
                        prob=min(1.0, t_prob + 0.25),
                        max_subs=max(1, t_subs + 1),
                        mismatch_prob=min(0.6, t_mis + 0.1),
                    )
                    with torch.no_grad():
                        txt_z_adv_t2i = encode_with_prompt_ensemble(
                            mm, picked_trades, tpl_train, device
                        )

                    loss_trades = cross_modal_trades(
                        i_clean=img_z,
                        t_clean=txt_z,
                        i_adv=img_z_adv_i2t,
                        t_adv=txt_z_adv_t2i,
                        logit_scale=mm.get_logit_scale(),
                        T=trades_T,
                        weight=1.0,
                    ) * trades_w

                    logits_clean_i2t = mm.get_logit_scale() * (
                        F.normalize(img_z, dim=-1) @ F.normalize(txt_z, dim=-1).T
                    )
                    logits_adv_i2t = mm.get_logit_scale() * (
                        F.normalize(img_z_adv_i2t, dim=-1) @ F.normalize(txt_z, dim=-1).T
                    )
                    loss_fap = flip_aware_margin(
                        logits_clean_i2t,
                        logits_adv_i2t.detach(),
                        margin=fap_margin,
                        weight=fap_w,
                    )

                    txtA, txtB = make_two_text_views(variant_embs)
                    if txtA is None or txtB is None or txtA.size(0) != img_z.size(0):
                        txtA = encode_with_prompt_ensemble(mm, picked, tpl_train, device)
                        txtB = encode_with_prompt_ensemble(mm, picked, tpl_train, device)

                    if trades_i2i_w > 0.0:
                        imgs_quick_i2i = quick_adv_i2i_images(
                            mm, imgs, img_pos_img, cfg,
                            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                            objective="away",
                        )
                        with torch.no_grad():
                            img_z_adv_i2i = mm.encode_image(imgs_quick_i2i)
                        loss_trades_i2i = trades_i2i(
                            i_clean=img_z,
                            i_gallery_clean=img_pos,
                            i_adv=img_z_adv_i2i,
                            logit_scale=mm.get_logit_scale(),
                            T=trades_T,
                            weight=1.0,
                        ) * trades_i2i_w
                    else:
                        loss_trades_i2i = torch.tensor(0.0, device=device)

                    if w_label_i2i > 0.0 and img_pos is not None:
                        la_i2i = label_aware_i2i(
                            img_q=img_z,
                            img_k=img_pos,
                            labels=labels,
                            logit_scale=mm.get_logit_scale(),
                            assume_normalized=True,
                        )
                    else:
                        la_i2i = torch.tensor(0.0, device=device)

                    if w_label_t2t_core > 0.0:
                        if txtA is None or txtB is None or txtA.size(0) != img_z.size(0):
                            txtA = encode_with_prompt_ensemble(mm, picked, tpl_train, device)
                            txtB = encode_with_prompt_ensemble(mm, picked, tpl_train, device)
                        la_t2t_core = label_aware_t2t_core(
                            txt_a=txtA,
                            txt_b=txtB,
                            labels=labels,
                            logit_scale=mm.get_logit_scale(),
                            assume_normalized=True,
                        )
                    else:
                        la_t2t_core = torch.tensor(0.0, device=device)

                    if mass_enabled:
                        mass_out = mass_losses_light(
                            img_clean=img_z,
                            txt_clean=txt_z,
                            logit_scale=mm.get_logit_scale(),
                            K=int(mass_cfg.get("K", 3)),
                            base_T=float(mass_cfg.get("base_T", trades_T)),
                            alpha=float(mass_cfg.get("alpha", 0.25)),
                            smin=float(mass_cfg.get("sigma_min", 0.04)),
                            smax=float(mass_cfg.get("sigma_max", 0.25)),
                            m_ref=float(mass_cfg.get("m_ref", 0.25)),
                            beta=float(mass_cfg.get("beta", 0.6)),
                            T_max_factor=float(mass_cfg.get("T_max_factor", 1.5)),
                            assume_normalized=True,
                        )
                        loss_mass_trades = mass_w_trades * (
                            mass_out["trades_smooth_i2t"] + mass_out["trades_smooth_t2i"]
                        )
                        loss_mass_cons = mass_w_cons * (
                            mass_out["cons_i2t"] + mass_out["cons_t2i"]
                        )
                        loss_mass_proto = torch.tensor(0.0, device=device)
                    else:
                        loss_mass_trades = torch.tensor(0.0, device=device)
                        loss_mass_cons = torch.tensor(0.0, device=device)
                        loss_mass_proto = torch.tensor(0.0, device=device)

                    la_clip_xm = label_aware_clip_xmodal(
                        img_z,
                        txt_z,
                        labels=labels,
                        logit_scale=mm.get_logit_scale(),
                        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
                        pair_boost=float(loss_cfg.get("la_pair_boost", 0.2)),
                        assume_normalized=True,
                    )
                    la_clip_xm = torch.clamp(la_clip_xm, 0.0, 50.0)

                    la_para_t2t = label_aware_paraphrase_consensus(
                        variant_embs,
                        labels=labels,
                        intra_w=loss_cfg.get("la_t2t_intra_w", 1.0),
                        inter_w=loss_cfg.get("la_t2t_inter_w", 1.0),
                        margin=loss_cfg.get("la_t2t_margin", 0.15),
                        assume_normalized=True,
                    )

                    if isinstance(blend, torch.Tensor):
                        sbi_src = img_pos
                    elif isinstance(imgs2, torch.Tensor):
                        sbi_src = mm.encode_image(imgs2)
                    else:
                        sbi_src = None

                    loss_sbi_xlbl = sbi_cross_label_loss(
                        img_clean=img_z,
                        img_sbi=sbi_src,
                        labels=labels,
                        mode=loss_cfg.get("sbi_mode", "auto"),
                        pull_margin=loss_cfg.get("sbi_pull_margin", 0.15),
                        push_margin=loss_cfg.get("sbi_push_margin", 0.15),
                        weight_pull=loss_cfg.get("sbi_weight_pull", 1.0),
                        weight_push=loss_cfg.get("sbi_weight_push", 1.0),
                        assume_normalized=True,
                    )

                    picked_trades_A = _strong_text_perturb(
                        picked_clean,
                        prob=min(1.0, t_prob + 0.50),
                        max_subs=max(2, t_subs + 2),
                        mismatch_prob=min(0.8, t_mis + 0.30),
                    )
                    with torch.no_grad():
                        txtA_adv = encode_with_prompt_ensemble(
                            mm, picked_trades_A, tpl_train, device
                        )

                    loss_trades_t2t = trades_t2t(
                        tA_clean=txtA,
                        tB_clean=txtB,
                        tA_adv=txtA_adv,
                        tB_adv=txtB,
                        logit_scale=mm.get_logit_scale(),
                        T=trades_T,
                        weight=1.0,
                    ) * trades_t2t_w

                    align_w = float(robust_cfg.get("align_w", 0.1))
                    if align_w > 0.0:
                        loss_robust_align = robust_alignment_loss(
                            clean_emb=img_z,
                            adv_emb=img_z_adv_i2t,
                            weight=align_w,
                            assume_normalized=True,
                        )
                    else:
                        loss_robust_align = torch.tensor(0.0, device=device)

                    forgery_proto_w = float(loss_cfg.get("forgery_proto_w", 0.0))
                    forgery_num_classes = int(loss_cfg.get("forgery_num_classes", 0))
                    if forgery_proto_w > 0.0 and forgery_num_classes > 0:
                        loss_forgery_proto = forgery_proto_loss(
                            emb=img_z,
                            labels=labels,
                            num_classes=forgery_num_classes,
                            weight=forgery_proto_w,
                            assume_normalized=True,
                        )
                    else:
                        loss_forgery_proto = torch.tensor(0.0, device=device)

                    total = (
                        w_it * loss_it
                        + w_tt * loss_tt
                        + w_ii * loss_ii
                        + loss_ic
                        + loss_temp
                        + loss_pos
                        + w_text_only * loss_txt_only
                        + loss_trades
                        + loss_fap
                        + loss_trades_t2t
                        + loss_trades_i2i
                        + loss_mass_trades
                        + loss_mass_cons
                        + loss_mass_proto
                        + w_label_clip_xm * la_clip_xm
                        + w_label_para_t2t * la_para_t2t
                        + w_sbi_xlbl * loss_sbi_xlbl
                        + w_label_i2i * la_i2i
                        + w_label_t2t_core * la_t2t_core
                        + loss_robust_align
                        + loss_forgery_proto
                    )

                    return total, {
                        "it": loss_it,
                        "txt_only": loss_txt_only,
                        "tt": loss_tt,
                        "ii": loss_ii,
                        "ic": loss_ic,
                        "pos": loss_pos,
                        "temp": loss_temp,
                        "trades": loss_trades,
                        "fap": loss_fap,
                        "t2t": loss_trades_t2t,
                        "i2i_trades": loss_trades_i2i,
                        "mass_trades": loss_mass_trades,
                        "mass_cons": loss_mass_cons,
                        "mass_proto": loss_mass_proto,
                        "la_clip_xm": la_clip_xm,
                        "la_para_t2t": la_para_t2t,
                        "la_sbi": loss_sbi_xlbl,
                        "la_i2i": la_i2i,
                        "la_t2t_core": la_t2t_core,
                        "robust_align": loss_robust_align,
                        "forgery_proto": loss_forgery_proto,
                    }

            total, part_dict = compute_total_and_parts()

            if not torch.isfinite(total):
                optimizer.zero_grad(set_to_none=True)
                tqdm.write("[StepSummary] non-finite loss; skip", end="\n")
                continue

            train_sum["total"] += float(total.item())
            for k in [
                "it","txt_only","tt","ii","ic","pos","temp",
                "trades","fap","t2t","i2i_trades",
                "mass_trades","mass_cons","mass_proto",
                "la_clip_xm","la_para_t2t","la_sbi",
                "la_i2i","la_t2t_core",
                "robust_align","forgery_proto",
            ]:
                train_sum[k] += float(part_dict[k].item())

            train_count += 1

            if (nstep % 50) == 0:
                tqdm.write(
                    f"[StepSummary] ep={epoch_abs} "
                    f"step={nstep}/{len(train_loader)} "
                    f"tot={total.item():.2f} "
                    f"it={part_dict['it'].item():.2f} "
                    f"tt={part_dict['tt'].item():.2f}",
                    end="\n",
                )

            loss = total / max(1, accum_steps)

            if use_awp and ((nstep % accum_steps) == 0):
                if amp_mode == "fp16":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                awp.perturb()
                optimizer.zero_grad(set_to_none=True)

                total_re, _ = compute_total_and_parts()
                loss_re = total_re / max(1, accum_steps)
                if amp_mode == "fp16":
                    scaler.scale(loss_re).backward()
                else:
                    loss_re.backward()
            else:
                if amp_mode == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            nstep += 1
            run_state["global_step"] += 1

            if (run_state["global_step"] % 100) == 0:
                tot = float(total.detach().item())
                ratios = {}
                for k in ["it","txt_only","tt","ii","trades","t2t","i2i_trades",
                          "mass_trades","mass_cons", "la_i2i","la_t2t_core", "robust_align","forgery_proto","fap"]:
                    v = float(part_dict[k].detach().item())
                    ratios[k] = (v / tot) if tot > 0 else 0.0
                print("LOSS RATIOS:", ratios, flush=True)

            if (nstep % accum_steps) == 0:
                if max_grad_norm > 0:
                    if amp_mode == "fp16":
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(mm.parameters(), max_grad_norm)

                if use_awp:
                    awp.restore()

                if amp_mode == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                clamp_logit_scale_if_present(
                    model,
                    loss_cfg.get("logit_scale_min", 10.0),
                    loss_cfg.get("logit_scale_max", 100.0),
                )

                ema.update(mm)

        train_mean = {
            k: (v / train_count if train_count > 0 else 0.0)
            for k, v in train_sum.items()
        }

        tgt = mm
        bak = {n: p.detach().clone() for n, p in tgt.named_parameters()}
        ema.copy_to(tgt)
        model.eval()

        vt_it = vt_tt = vt_ic = vt_ii = 0.0
        vt_pos = vt_temp = 0.0
        vt_txt_only = 0.0
        vt_forgery_proto = 0.0

        vc = 0
        vt_la_clip_xm = 0.0
        vt_la_para_t2t = 0.0
        vt_la_sbi = 0.0
        vt_la_i2i = 0.0
        vt_la_t2t_core = 0.0

        R1_i2t = R5_i2t = MR_i2t = MRR_i2t = NDCG5_i2t = AUCMC_i2t = AUROC_i2t = 0.0
        R1_t2i = R5_t2i = MR_t2i = MRR_t2i = NDCG5_t2i = AUCMC_t2i = AUROC_t2i = 0.0
        R1_i2i = R5_i2i = MR_i2i = MRR_i2i = NDCG5_i2i = AUCMC_i2i = AUROC_i2i = 0.0
        R1_t2t = R5_t2t = MR_t2t = MRR_t2t = NDCG5_t2t = AUCMC_t2t = AUROC_t2t = 0.0

        R1_rob = R5_rob = MR_rob = AUROC_rob = 0.0
        robust_batches = 0
        val_has_blend = 0

        loss_cfg = cfg.get("loss", {})
        robust_cfg = cfg.get("robust", {})

        with torch.no_grad():
            vbar = tqdm(
                val_loader,
                desc=f"V{epoch_abs}",
                position=0,
                leave=True,
                ncols=60,
                dynamic_ncols=False,
                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                mininterval=1.0,
                smoothing=0,
            )
            for batch in vbar:
                imgs = batch["image"].to(device, non_blocking=True)
                imgs2 = batch["image2"]
                imgs2 = imgs2.to(device, non_blocking=True) if isinstance(imgs2, torch.Tensor) else None
                blend = batch["blend_image"]
                blend = blend.to(device, non_blocking=True) if isinstance(blend, torch.Tensor) else None
                texts_list = batch["texts"]
                labels = batch["label"].to(device)

                picked = build_text_batch_for_it_balanced(texts_list, step=0)

                img_z_raw = mm.encode_image(imgs)
                txt_z_raw = encode_with_prompt_ensemble(mm, picked, tpl_eval, device)

                refiner = getattr(mm, "refine_embeddings", None)
                if callable(refiner):
                    img_z, txt_z = refiner(img_z_raw, txt_z_raw, imgs)
                else:
                    img_z, txt_z = img_z_raw, txt_z_raw

                vt_it += clip_like_loss(
                    img_z, txt_z, mm.get_logit_scale(),
                    label_smoothing, assume_normalized=True,
                ).item()

                variant_embs = encode_variants_with_prompts(
                    mm, texts_list, tpl_eval, device
                )
                vt_tt += paraphrase_consensus_loss(
                    variant_embs, mm.get_logit_scale(),
                    assume_normalized=True,
                ).item()

                if pos_align_w > 0:
                    vt_pos += (
                        (
                            1.0 - (
                                F.normalize(img_z, dim=-1)
                                * F.normalize(txt_z, dim=-1)
                            ).sum(dim=1)
                        ).mean() * pos_align_w
                    ).item()
                vt_temp += temperature_reg(
                    mm.get_logit_scale(),
                    target=float(loss_cfg.get("temp_target", 100.0)),
                    weight=float(loss_cfg.get("temp_reg", 5.0e-4)),
                ).item()

                w_text_only = float(loss_cfg.get("w_text_only", 0.0))
                if w_text_only > 0.0:
                    txt_only_val = text_only_contrastive_loss(
                        txt_z=txt_z,
                        temperature=float(loss_cfg.get("text_only_T", 2.0)),
                        assume_normalized=True,
                    )
                    vt_txt_only += txt_only_val.item()
                else:
                    txt_only_val = None

                la_clip_xm_val = label_aware_clip_xmodal(
                    img_z,
                    txt_z,
                    labels=labels,
                    logit_scale=mm.get_logit_scale(),
                    label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
                    pair_boost=float(loss_cfg.get("la_pair_boost", 0.2)),
                    assume_normalized=True,
                )

                la_para_t2t_val = label_aware_paraphrase_consensus(
                    variant_embs,
                    labels=labels,
                    intra_w=float(loss_cfg.get("la_t2t_intra_w", 1.0)),
                    inter_w=float(loss_cfg.get("la_t2t_inter_w", 1.0)),
                    margin=float(loss_cfg.get("la_t2t_margin", 0.15)),
                    assume_normalized=True,
                )

                txtA, txtB = make_two_text_views(variant_embs)
                if txtA is None or txtB is None or txtA.size(0) != img_z.size(0):
                    txtA = encode_with_prompt_ensemble(mm, picked, tpl_eval, device)
                    txtB = encode_with_prompt_ensemble(mm, picked, tpl_eval, device)
                sim_tt = cosine_sim(txtA, txtB)
                R1_t2t += r_at_1_from_sim(sim_tt)
                R5_t2t += r_at_5_from_sim(sim_tt)
                MR_t2t += median_rank(sim_tt)
                MRR_t2t += mean_reciprocal_rank(sim_tt)
                NDCG5_t2t += ndcg_at_k(sim_tt, 5)
                AUCMC_t2t += auc_cmc(sim_tt)
                AUROC_t2t += pairwise_auroc(sim_tt)

                img_pos = None
                blend_z = None
                imgs2_z = None

                if isinstance(blend, torch.Tensor):
                    val_has_blend += 1
                    blend_z = mm.encode_image(blend)
                    img_pos = blend_z
                    vt_ic += image_consistency_loss(
                        img_z, blend_z,
                        weight=loss_cfg.get("img_consistency", 0.1),
                        assume_normalized=True,
                    ).item()
                elif isinstance(imgs2, torch.Tensor):
                    imgs2_z = mm.encode_image(imgs2)
                    img_pos = imgs2_z

                sbi_val = sbi_cross_label_loss(
                    img_clean=img_z,
                    img_sbi=(blend_z if blend_z is not None else imgs2_z),
                    labels=labels,
                    mode=loss_cfg.get("sbi_mode", "auto"),
                    pull_margin=float(loss_cfg.get("sbi_pull_margin", 0.15)),
                    push_margin=float(loss_cfg.get("sbi_push_margin", 0.15)),
                    weight_pull=float(loss_cfg.get("sbi_weight_pull", 1.0)),
                    weight_push=float(loss_cfg.get("sbi_weight_push", 1.0)),
                    assume_normalized=True,
                )

                vt_la_clip_xm += la_clip_xm_val.item()
                vt_la_para_t2t += la_para_t2t_val.item()
                vt_la_sbi += sbi_val.item()

                w_label_i2i = float(loss_cfg.get("w_label_i2i", 0.0))
                if w_label_i2i > 0.0 and img_pos is not None:
                    la_i2i_val = label_aware_i2i(
                        img_q=img_z,
                        img_k=img_pos,
                        labels=labels,
                        logit_scale=mm.get_logit_scale(),
                        assume_normalized=True,
                    )
                    vt_la_i2i += la_i2i_val.item()

                w_label_t2t_core = float(loss_cfg.get("w_label_t2t_core", 0.0))
                if w_label_t2t_core > 0.0:
                    la_t2t_core_val = label_aware_t2t_core(
                        txt_a=txtA,
                        txt_b=txtB,
                        labels=labels,
                        logit_scale=mm.get_logit_scale(),
                        assume_normalized=True,
                    )
                    vt_la_t2t_core += la_t2t_core_val.item()

                forgery_proto_w = float(loss_cfg.get("forgery_proto_w", 0.0))
                forgery_num_classes = int(loss_cfg.get("forgery_num_classes", 0))
                if forgery_proto_w > 0.0 and forgery_num_classes > 0:
                    fp_val = forgery_proto_loss(
                        emb=img_z,
                        labels=labels,
                        num_classes=forgery_num_classes,
                        weight=forgery_proto_w,
                        assume_normalized=True,
                    )
                    vt_forgery_proto += fp_val.item()

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
                    vt_ii += clip_like_loss(
                        img_z, img_pos, mm.get_logit_scale(),
                        label_smoothing, assume_normalized=True,
                    ).item()

                    sim_ii = cosine_sim(img_z, img_pos)
                    R1_i2i += r_at_1_from_sim(sim_ii)
                    R5_i2i += r_at_5_from_sim(sim_ii)
                    MR_i2i += median_rank(sim_ii)
                    MRR_i2i += mean_reciprocal_rank(sim_ii)
                    NDCG5_i2i += ndcg_at_k(sim_ii, 5)
                    AUCMC_i2i += auc_cmc(sim_ii)
                    AUROC_i2i += pairwise_auroc(sim_ii)

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
        val_it = vt_it / max(vc, 1)
        val_tt = vt_tt / max(vc, 1)
        val_ic = vt_ic / max(val_has_blend, 1) if val_has_blend > 0 else 0.0
        val_ii = vt_ii / max(vc, 1) if vt_ii > 0 else 0.0
        val_pos = vt_pos / max(vc, 1)
        val_temp = vt_temp / max(vc, 1)
        val_txt_only = vt_txt_only / max(vc, 1) if vt_txt_only > 0 else 0.0
        val_forgery_proto = vt_forgery_proto / max(vc, 1) if vt_forgery_proto > 0 else 0.0

        logdict = {
            "val/it_loss": val_it,
            "val/tt_loss": val_tt,
            "val/ii_loss": val_ii,
            "val/ic_loss": val_ic,
            "val/pos": val_pos,
            "val/temp": val_temp,
            "val/text_only_loss": val_txt_only,
            "val/forgery_proto_loss": val_forgery_proto,

            "val/label_clip_xm_loss": (vt_la_clip_xm / max(vc, 1)) * float(loss_cfg.get("w_label_clip_xm", 0.0)),
            "val/label_para_t2t_loss": (vt_la_para_t2t / max(vc, 1)) * float(loss_cfg.get("w_label_para_t2t", 0.0)),
            "val/label_sbi_loss": (vt_la_sbi / max(vc, 1)) * float(loss_cfg.get("w_sbi_xlbl", 0.0)),
            "val/label_i2i_loss": (vt_la_i2i / max(vc, 1)) * float(loss_cfg.get("w_label_i2i", 0.0)),
            "val/label_t2t_core_loss": (vt_la_t2t_core / max(vc, 1)) * float(loss_cfg.get("w_label_t2t_core", 0.0)),

            "val/R@1_i2t": safe_div(R1_i2t), "val/R@5_i2t": safe_div(R5_i2t),
            "val/MR_i2t": safe_div(MR_i2t), "val/MRR_i2t": safe_div(MRR_i2t),
            "val/NDCG@5_i2t": safe_div(NDCG5_i2t),
            "val/AUCMC_i2t": safe_div(AUCMC_i2t), "val/AUROC_i2t": safe_div(AUROC_i2t),

            "val/R@1_t2i": safe_div(R1_t2i), "val/R@5_t2i": safe_div(R5_t2i),
            "val/MR_t2i": safe_div(MR_t2i), "val/MRR_t2i": safe_div(MRR_t2i),
            "val/NDCG@5_t2i": safe_div(NDCG5_t2i),
            "val/AUCMC_t2i": safe_div(AUCMC_t2i), "val/AUROC_t2i": safe_div(AUROC_t2i),

            "val/R@1_i2i": (safe_div(R1_i2i) if vt_ii > 0 else 0.0),
            "val/R@5_i2i": (safe_div(R5_i2i) if vt_ii > 0 else 0.0),
            "val/MR_i2i": (safe_div(MR_i2i) if vt_ii > 0 else 0.0),
            "val/MRR_i2i": (safe_div(MRR_i2i) if vt_ii > 0 else 0.0),
            "val/NDCG@5_i2i": (safe_div(NDCG5_i2i) if vt_ii > 0 else 0.0),
            "val/AUCMC_i2i": (safe_div(AUCMC_i2i) if vt_ii > 0 else 0.0),
            "val/AUROC_i2i": (safe_div(AUROC_i2i) if vt_ii > 0 else 0.0),

            "val/R@1_t2t": safe_div(R1_t2t),
            "val/R@5_t2t": safe_div(R5_t2t),
        }

        avg_la_clip = vt_la_clip_xm / max(vc, 1)
        if avg_la_clip > 50.0:
            logdict["val/label_clip_xm_loss_raw"] = avg_la_clip

        avg_la_sbi = vt_la_sbi / max(vc, 1)
        if avg_la_sbi > 5.0:
            logdict["val/label_sbi_loss_raw"] = avg_la_sbi

        try:
            del imgs, imgs2, blend, texts_list, img_z, txt_z, img_z_raw, txt_z_raw
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for n, p in tgt.named_parameters():
                if n in bak:
                    p.data.copy_(bak[n].data)
        del bak
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        full_ckpt = _abs(os.path.join(ckpt_dir, f"{arch_tag}_mm_epochABS{epoch_abs}_FULL.pt"))
        try:
            save_full_ckpt(
                full_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                cfg=cfg,
                epoch=epoch_abs,
                global_step=run_state["global_step"],
                best_val=best_val,
                best_rob=best_rob,
                bad=bad,
                bad_rob=bad_rob,
            )
        except Exception:
            pass

        run_eval_adv = should_run_eval_adv(cfg, epoch_rel)
        if run_eval_adv:
            txt_cfg = cfg.get("eval_adv", {}).get("text", {})
            tprob_eval = float(txt_cfg.get("prob", 0.0)) if bool(txt_cfg.get("enable", False)) else 0.0
            text_cfg_eval = {
                "prob": tprob_eval,
                "max_subs": int(ttxt_cfg.get("max_subs", 2)),
                "enable": bool(txt_cfg.get("enable", False)),
            }

            max_adv_val_batches = int(cfg.get("eval_adv", {}).get("max_val_batches", 64))
            from itertools import islice
            limited_val_iter2 = islice(val_loader, max_adv_val_batches)

            print(f"[EvalAdv] Starting adversarial evaluation at epoch {epoch_abs} ...", flush=True)

            adv_all_metrics = eval_adv_all(
                mm,
                limited_val_iter2,
                device,
                tpl_eval,
                cfg.get("eval_adv", {}).get("pgd", {}),
                text_cfg_eval,
                cfg,
                total=max_adv_val_batches,
                desc=f"Adv{epoch_abs}",
            )

            for k, v in adv_all_metrics.items():
                logdict[f"val/{k}"] = v
            logdict["val/eval_adv_ran"] = 1.0

            print(f"[EvalAdv] Finished epoch {epoch_abs}", flush=True)
        else:
            logdict["val/eval_adv_ran"] = 0.0

        print(
            "[TrainSummary] "
            f"epoch_abs={epoch_abs} "
            f"loss={train_mean['total']:.4f} | "
            f"it={train_mean['it']:.4f} tt={train_mean['tt']:.4f} ii={train_mean['ii']:.4f} ic={train_mean['ic']:.4f} | "
            f"pos={train_mean['pos']:.4f} temp={train_mean['temp']:.4f} txt_only={train_mean.get('txt_only', 0.0):.4f} "
            f"forgery={train_mean.get('forgery_proto', 0.0):.4f} | "
            f"trades={train_mean['trades']:.4f} t2t={train_mean['t2t']:.4f} fap={train_mean['fap']:.4f} i2i_trades={train_mean['i2i_trades']:.4f} | "
            f"MASS trades/cons/proto={train_mean['mass_trades']:.4f}/{train_mean['mass_cons']:.4f}/{train_mean['mass_proto']:.4f} "
            f"LA clip_xm/para_t2t/sbi={train_mean['la_clip_xm']:.4f}/{train_mean['la_para_t2t']:.4f}/{train_mean['la_sbi']:.4f} "
            f"rob_align={train_mean.get('robust_align', 0.0):.4f}",
            flush=True
        )

        print(
            "[ValSummary] "
            f"epoch_abs={epoch_abs} rel={epoch_rel} | "
            f"loss_it={logdict['val/it_loss']:.4f} loss_tt={logdict['val/tt_loss']:.4f} "
            f"loss_ii={logdict['val/ii_loss']:.4f} loss_ic={logdict['val/ic_loss']:.4f} "
            f"| pos={logdict['val/pos']:.4f} temp={logdict['val/temp']:.4f} "
            f"text_only={logdict['val/text_only_loss']:.4f} forgery={logdict['val/forgery_proto_loss']:.4f} | "
            f"label_clip_xm={logdict['val/label_clip_xm_loss']:.4f} "
            f"label_para_t2t={logdict['val/label_para_t2t_loss']:.4f} "
            f"label_sbi={logdict['val/label_sbi_loss']:.4f} "
            f"I2T R@1/5={logdict['val/R@1_i2t']:.3f}/{logdict['val/R@5_i2t']:.3f}  "
            f"T2I R@1/5={logdict['val/R@1_t2i']:.3f}/{logdict['val/R@5_t2i']:.3f}  "
            f"I2I R@1/5={logdict['val/R@1_i2i']:.3f}/{logdict['val/R@5_i2i']:.3f}  "
            f"T2T R@1/5={logdict['val/R@1_t2t']:.3f}/{logdict['val/R@5_t2t']:.3f}",
            flush=True
        )

        if logdict.get("val/eval_adv_ran", 0.0) == 1.0:
            print(
                "[ValAdvSummary] "
                f"epoch_abs={epoch_abs} rel={epoch_rel} | "
                f"ADV R@1 i2t/t2i/i2i/t2t="
                f"{logdict.get('val/R@1_i2t_adv', float('nan')):.3f}/"
                f"{logdict.get('val/R@1_t2i_adv', float('nan')):.3f}/"
                f"{logdict.get('val/R@1_i2i_adv', float('nan')):.3f}/"
                f"{logdict.get('val/R@1_t2t_adv', float('nan')):.3f}",
                flush=True
            )

        log_epoch = {
            **{f"train/{k}": v for k, v in train_mean.items()},
            **logdict,
            "epoch_abs": epoch_abs,
            "epoch_rel": epoch_rel,
            "step": run_state["global_step"],
        }
        logger.log(log_epoch)

        if enable_early_stop:
            val_comp = 0.5 * (1.0 - logdict["val/R@1_i2t"]) + 0.5 * (1.0 - logdict["val/R@1_t2i"])
            if best_val is None or (val_comp < best_val):
                best_val = val_comp
                bad = 0
            else:
                bad += 1
                if bad >= int(cfg["train"].get("patience", 12)):
                    break

            robust_key = cfg["loss"].get("robust_metric_key", "val/R@1_i2t_adv")
            if robust_key in logdict and not math.isnan(logdict[robust_key]):
                cur_rob = logdict[robust_key]
                if (best_rob is None) or (cur_rob > best_rob + 1e-6):
                    best_rob = cur_rob
                    bad_rob = 0
                else:
                    bad_rob += 1
                    if bad_rob >= int(cfg["loss"].get("robust_plateau_patience", 3)):
                        break

            run_state["best_val"] = best_val
            run_state["bad"] = bad
            run_state["best_rob"] = best_rob
            run_state["bad_rob"] = bad_rob

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_full = _abs(os.path.join(ckpt_dir, f"{arch_tag}_FINAL_FULL.pt"))
    try:
        save_full_ckpt(
            final_full,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            cfg=cfg,
            epoch=max(1, run_state["epoch"]),
            global_step=run_state["global_step"],
            best_val=run_state["best_val"],
            best_rob=run_state["best_rob"],
            bad=run_state["bad"],
            bad_rob=run_state["bad_rob"],
        )
    except Exception:
        pass

    print("[DONE]", flush=True)
    logger.finish()

if __name__ == "__main__":
    main()