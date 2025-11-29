#scripts/losses_mm.py

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any

# ========= basic utils =========

def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def sim_matrix(a: torch.Tensor, b: torch.Tensor, assume_normalized: bool = True) -> torch.Tensor:
    if not assume_normalized:
        a = _l2norm(a)
        b = _l2norm(b)
    return a @ b.t()

def sim_logits(a: torch.Tensor, b: torch.Tensor, logit_scale: torch.Tensor, assume_normalized: bool = True) -> torch.Tensor:
    return float(logit_scale) * sim_matrix(a, b, assume_normalized=assume_normalized)

# ========= classic CLIP-like (instance-level) =========

def clip_like_loss(
    img_z: torch.Tensor,
    txt_z: torch.Tensor,
    logit_scale: torch.Tensor,
    label_smoothing: float = 0.0,
    assume_normalized: bool = True,
) -> torch.Tensor:
    logits_i2t = sim_logits(img_z, txt_z, logit_scale, assume_normalized=assume_normalized)
    logits_t2i = logits_i2t.t()
    targets = torch.arange(img_z.size(0), device=img_z.device)
    li = F.cross_entropy(logits_i2t, targets, label_smoothing=float(label_smoothing))
    lt = F.cross_entropy(logits_t2i, targets, label_smoothing=float(label_smoothing))
    return 0.5 * (li + lt)

# ========= text-only contrastive (to help T2T) =========

def text_only_contrastive_loss(
    txt_z: torch.Tensor,
    temperature: float = 1.0,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if not assume_normalized:
        txt_z = _l2norm(txt_z)
    logits = (txt_z @ txt_z.t()) / float(temperature)
    B = txt_z.size(0)
    targets = torch.arange(B, device=txt_z.device)
    return F.cross_entropy(logits, targets)

# ========= label-aware CLIP (class-level, for forgery) =========

def label_aware_clip_xmodal(
    img_z: torch.Tensor,
    txt_z: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
    label_smoothing: float = 0.0,
    pair_boost: float = 0.2,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if not assume_normalized:
        img_z = _l2norm(img_z)
        txt_z = _l2norm(txt_z)

    logits = float(logit_scale) * (img_z @ txt_z.t())
    B = logits.size(0)
    dev = logits.device
    eye = torch.eye(B, device=dev)
    same_label = (labels.view(B, 1) == labels.view(1, B)).float()
    pos_mask = torch.maximum(same_label, eye)

    if pair_boost > 0.0:
        logits = logits + eye * float(pair_boost)

    masked_logits_i2t = logits + (1.0 - pos_mask) * (-1e4)
    targets = torch.arange(B, device=dev)
    loss_i2t = F.cross_entropy(masked_logits_i2t, targets, label_smoothing=float(label_smoothing))

    masked_logits_t2i = logits.t() + (1.0 - pos_mask.t()) * (-1e4)
    loss_t2i = F.cross_entropy(masked_logits_t2i, targets, label_smoothing=float(label_smoothing))

    return 0.5 * (loss_i2t + loss_t2i)

# ========= label-aware image<->image (for SBI / second view) =========

def label_aware_i2i(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if not assume_normalized:
        img_q = _l2norm(img_q)
        img_k = _l2norm(img_k)

    logits = float(logit_scale) * (img_q @ img_k.t())
    B = logits.size(0)
    dev = logits.device
    eye = torch.eye(B, device=dev)
    same_label = (labels.view(B, 1) == labels.view(1, B)).float()
    pos_mask = torch.maximum(same_label, eye)
    masked_logits = logits + (1.0 - pos_mask) * (-1e4)
    targets = torch.arange(B, device=dev)
    return F.cross_entropy(masked_logits, targets)

# ========= SBI cross-label branch =========

def sbi_cross_label_loss(
    img_clean: torch.Tensor,
    img_sbi: Optional[torch.Tensor],
    labels: torch.Tensor,
    mode: str = "auto",
    pull_margin: float = 0.15,
    push_margin: float = 0.15,
    weight_pull: float = 1.0,
    weight_push: float = 1.0,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if img_sbi is None:
        return torch.tensor(0.0, device=img_clean.device)

    if not assume_normalized:
        img_clean = _l2norm(img_clean)
        img_sbi = _l2norm(img_sbi)

    sims = (img_clean * img_sbi).sum(dim=-1)
    B = sims.size(0)

    if mode == "force_neg":
        push_loss = F.relu(sims - (1.0 - push_margin)).mean()
        return float(weight_push) * push_loss

    lbl = labels.view(B)
    pull_mask = torch.ones_like(lbl, dtype=torch.bool, device=lbl.device)
    sims_pull = sims[pull_mask]
    pull_loss = torch.tensor(0.0, device=img_clean.device)
    if sims_pull.numel() > 0:
        pull_loss = F.relu(pull_margin - sims_pull).mean()

    return float(weight_pull) * pull_loss

# ========= paraphrase / text clustering =========

def paraphrase_consensus_loss(
    txt_group_embs: List[Optional[torch.Tensor]],
    logit_scale: Optional[torch.Tensor] = None,
    assume_normalized: bool = True,
) -> torch.Tensor:
    loss = 0.0
    cnt = 0
    for emb in txt_group_embs:
        if emb is None or emb.size(0) < 2:
            continue
        sim = sim_matrix(emb, emb, assume_normalized=assume_normalized)
        mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        pos = sim[mask]
        loss = loss + (1.0 - pos).mean()
        cnt += 1
    if cnt == 0:
        dev = txt_group_embs[0].device if (txt_group_embs and isinstance(txt_group_embs[0], torch.Tensor)) else "cpu"
        return torch.zeros((), device=dev)
    return loss / float(cnt)

# ========= label-aware paraphrase (use label as well) =========

def label_aware_paraphrase_consensus(
    txt_group_embs: List[Optional[torch.Tensor]],
    labels: torch.Tensor,
    intra_w: float = 1.0,
    inter_w: float = 1.0,
    margin: float = 0.15,
    assume_normalized: bool = True,
) -> torch.Tensor:
    device = labels.device
    B = len(txt_group_embs)
    losses: List[torch.Tensor] = []

    for i in range(B):
        ve = txt_group_embs[i]
        if ve is None or ve.size(0) < 2:
            continue
        z = ve if assume_normalized else _l2norm(ve)
        center = z.mean(dim=0, keepdim=True)
        intra = 1.0 - (z * center).sum(dim=1)
        losses.append(float(intra_w) * intra.mean())

    centers: List[Optional[torch.Tensor]] = []
    for i in range(B):
        ve = txt_group_embs[i]
        if ve is None or ve.size(0) == 0:
            centers.append(None)
        else:
            z = ve if assume_normalized else _l2norm(ve)
            centers.append(z.mean(dim=0, keepdim=True))

    for i in range(B):
        ci = centers[i]
        if ci is None:
            continue
        yi = labels[i].item()
        for j in range(i + 1, B):
            cj = centers[j]
            if cj is None:
                continue
            yj = labels[j].item()
            sim = (ci * cj).sum()
            if yi == yj:
                losses.append(float(inter_w) * (1.0 - sim))
            else:
                diff = sim - (1.0 - margin)
                if diff > 0:
                    losses.append(float(inter_w) * diff)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()

# ========= temperature reg =========

def temperature_reg(logit_scale: torch.Tensor, target: float = 1.0/0.07, weight: float = 1e-3) -> torch.Tensor:
    tgt = torch.tensor(float(target), device=logit_scale.device, dtype=logit_scale.dtype)
    return float(weight) * (logit_scale - tgt).pow(2)

# ========= image consistency (SBI view staying close) =========

def image_consistency_loss(
    img_z: torch.Tensor,
    blend_z: Optional[torch.Tensor],
    weight: float = 0.2,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if blend_z is None:
        return torch.tensor(0.0, device=img_z.device)
    if not assume_normalized:
        img_z = _l2norm(img_z)
        blend_z = _l2norm(blend_z)
    sim = (img_z * blend_z).sum(-1)
    return float(weight) * (1.0 - sim).mean()

# ========= TRADES core utils =========

def _row_log_softmax(x: torch.Tensor, T: float = 1.0):
    x = x / float(T)
    return F.log_softmax(x, dim=1), F.softmax(x, dim=1)

def _kl_rows(p_log: torch.Tensor, p: torch.Tensor, q_log: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    kl_pq = F.kl_div(q_log, p, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(p_log, q, reduction="batchmean", log_target=False)
    return kl_pq + kl_qp

# ========= cross-modal TRADES =========

def cross_modal_trades(
    i_clean: torch.Tensor,
    t_clean: torch.Tensor,
    i_adv: torch.Tensor,
    t_adv: torch.Tensor,
    logit_scale: torch.Tensor,
    T: float = 1.0,
    weight: float = 1.0,
    assume_normalized: bool = True,
) -> torch.Tensor:
    Lc = sim_logits(i_clean, t_clean, logit_scale, assume_normalized=assume_normalized)
    La = sim_logits(i_adv,   t_clean, logit_scale, assume_normalized=assume_normalized)
    p_log_r, p_r = _row_log_softmax(Lc, T=T)
    q_log_r, q_r = _row_log_softmax(La, T=T)
    kl_i2t = _kl_rows(p_log_r, p_r, q_log_r, q_r)

    LcT = Lc.t()
    LaT = sim_logits(i_clean, t_adv, logit_scale, assume_normalized=assume_normalized).t()
    p_log_c, p_c = _row_log_softmax(LcT, T=T)
    q_log_c, q_c = _row_log_softmax(LaT, T=T)
    kl_t2i = _kl_rows(p_log_c, p_c, q_log_c, q_c)

    return float(weight) * 0.5 * (kl_i2t + kl_t2i)

# ========= flip-aware margin =========

def flip_aware_margin(
    logits_clean_i2t: torch.Tensor,
    logits_adv_i2t: torch.Tensor,
    margin: float = 0.10,
    weight: float = 1.0,
) -> torch.Tensor:
    with torch.no_grad():
        B = logits_clean_i2t.size(0)
        top_clean = logits_clean_i2t.argmax(dim=1)
        top_adv = logits_adv_i2t.argmax(dim=1)
        flipped = (top_clean != top_adv)
    if not flipped.any():
        return logits_clean_i2t.new_zeros(())
    rows = flipped.nonzero(as_tuple=False).squeeze(1)
    pos = logits_adv_i2t[rows, rows]
    row_logits = logits_adv_i2t[rows]
    mask = torch.ones_like(row_logits, dtype=torch.bool)
    mask[torch.arange(rows.numel(), device=rows.device), rows] = False
    neg_max = row_logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    hinge = F.relu(float(margin) + neg_max - pos).mean()
    return float(weight) * hinge

# ========= trades variants =========

def trades_i2i(
    i_clean: torch.Tensor,
    i_gallery_clean: torch.Tensor,
    i_adv: torch.Tensor,
    logit_scale: torch.Tensor,
    T: float = 1.0,
    weight: float = 1.0,
    assume_normalized: bool = True,
) -> torch.Tensor:
    Lc = sim_logits(i_clean, i_gallery_clean, logit_scale, assume_normalized=assume_normalized)
    La = sim_logits(i_adv,   i_gallery_clean, logit_scale, assume_normalized=assume_normalized)
    p_log, p = _row_log_softmax(Lc, T=T)
    q_log, q = _row_log_softmax(La, T=T)
    return float(weight) * _kl_rows(p_log, p, q_log, q)

def trades_t2t(
    tA_clean: torch.Tensor,
    tB_clean: torch.Tensor,
    tA_adv: torch.Tensor,
    tB_adv: torch.Tensor,
    logit_scale: torch.Tensor,
    T: float = 1.0,
    weight: float = 1.0,
    assume_normalized: bool = True,
) -> torch.Tensor:
    Lc = sim_logits(tA_clean, tB_clean, logit_scale, assume_normalized=assume_normalized)
    La = sim_logits(tA_adv,   tB_clean, logit_scale, assume_normalized=assume_normalized)
    p_log, p = _row_log_softmax(Lc, T=T)
    q_log, q = _row_log_softmax(La, T=T)
    return float(weight) * _kl_rows(p_log, p, q_log, q)

# ========= robust alignment (external suggestion) =========

def robust_alignment_loss(
    clean_emb: torch.Tensor,
    adv_emb: torch.Tensor,
    weight: float = 0.1,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if not assume_normalized:
        clean_emb = _l2norm(clean_emb)
        adv_emb = _l2norm(adv_emb)
    cs = F.cosine_similarity(clean_emb, adv_emb, dim=-1)
    return float(weight) * (1.0 - cs).mean()

# ========= simple forgery proto (ours, small) =========

def forgery_proto_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weight: float = 0.05,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if not assume_normalized:
        emb = _l2norm(emb)
    dev = emb.device
    D = emb.size(1)
    protos = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            proto = emb[mask].mean(dim=0, keepdim=True)
            proto = F.normalize(proto, dim=-1)
            protos.append(proto)
        else:
            protos.append(torch.zeros(1, D, device=dev))
    protos = torch.cat(protos, dim=0)
    proto_for_sample = protos[labels]
    cs = (emb * proto_for_sample).sum(dim=-1)
    return float(weight) * (1.0 - cs).mean()

# ========= MASS (lighter, fixed scaling) =========

def _pairwise_margin(sim_bb: torch.Tensor, diag_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B = sim_bb.size(0)
    ar = torch.arange(B, device=sim_bb.device)
    s_pos = sim_bb[ar, diag_index]
    sim_mask = sim_bb.clone()
    sim_mask[ar, diag_index] = -1e9
    s_neg = sim_mask.max(dim=1).values
    return (s_pos - s_neg).clamp(min=0.0), s_pos

def _adaptive_sigma(margin: torch.Tensor, alpha: float = 0.3, smin: float = 0.02, smax: float = 0.35, eps: float = 1e-6) -> torch.Tensor:
    sigma = alpha / (margin + eps)
    return sigma.clamp_(smin, smax)

def _eot_embeds(e_clean: torch.Tensor, sigma: torch.Tensor, K: int = 4) -> torch.Tensor:
    B, D = e_clean.shape
    noise = torch.randn(K, B, D, device=e_clean.device, dtype=e_clean.dtype)
    e_noisy = F.normalize(e_clean.unsqueeze(0) + noise * sigma.view(1, B, 1), dim=-1)
    return e_noisy

def _kl_mean_pairwise(dists: List[torch.Tensor]) -> torch.Tensor:
    K = len(dists)
    if K <= 1:
        return dists[0].new_zeros(())
    tot = 0.0
    cnt = 0
    for i in range(K):
        for j in range(i + 1, K):
            di, dj = dists[i], dists[j]
            tot = tot + F.kl_div(dj.log(), di, reduction="batchmean", log_target=True)
            tot = tot + F.kl_div(di.log(), dj, reduction="batchmean", log_target=True)
            cnt += 2
    return tot / max(cnt, 1)

def mass_losses_light(
    img_clean: torch.Tensor,
    txt_clean: torch.Tensor,
    logit_scale: torch.Tensor,
    K: int = 4,
    base_T: float = 1.0,
    alpha: float = 0.3,
    smin: float = 0.02,
    smax: float = 0.35,
    m_ref: float = 0.30,
    beta: float = 0.7,
    T_max_factor: float = 2.0,
    assume_normalized: bool = True,
) -> Dict[str, torch.Tensor]:
    if not assume_normalized:
        img_clean = _l2norm(img_clean)
        txt_clean = _l2norm(txt_clean)

    B = img_clean.size(0)
    diag = torch.arange(B, device=img_clean.device)

    S_i2t = img_clean @ txt_clean.t()
    S_t2i = S_i2t.t()
    m_i2t, _ = _pairwise_margin(S_i2t, diag)
    m_t2i, _ = _pairwise_margin(S_t2i, diag)

    sig_img = _adaptive_sigma(m_i2t, alpha=alpha, smin=smin, smax=smax)
    sig_txt = _adaptive_sigma(m_t2i, alpha=alpha, smin=smin, smax=smax)

    E_img = _eot_embeds(img_clean, sig_img, K=K)
    E_txt = _eot_embeds(txt_clean, sig_txt, K=K)

    T_i2t = base_T * (1.0 + beta * torch.relu(m_ref - m_i2t))
    T_t2i = base_T * (1.0 + beta * torch.relu(m_ref - m_t2i))
    T_i2t = torch.clamp(T_i2t, min=base_T, max=base_T * T_max_factor)
    T_t2i = torch.clamp(T_t2i, min=base_T, max=base_T * T_max_factor)

    Lc_i2t = float(logit_scale) * S_i2t
    Lc_t2i = float(logit_scale) * S_t2i
    p_clean_i2t = torch.softmax(Lc_i2t / base_T, dim=1)
    p_clean_t2i = torch.softmax(Lc_t2i / base_T, dim=1)

    dists_i2t: List[torch.Tensor] = []
    dists_t2i: List[torch.Tensor] = []
    for k in range(K):
        L_i2t = float(logit_scale) * (E_img[k] @ txt_clean.t())
        L_t2i = float(logit_scale) * (E_txt[k] @ img_clean.t())
        dists_i2t.append(torch.softmax(L_i2t / T_i2t.view(-1, 1), dim=1))
        dists_t2i.append(torch.softmax(L_t2i / T_t2i.view(-1, 1), dim=1))

    p_smooth_i2t = torch.stack(dists_i2t, dim=0).mean(dim=0)
    p_smooth_t2i = torch.stack(dists_t2i, dim=0).mean(dim=0)

    loss_trades_smooth_i2t = F.kl_div(p_smooth_i2t.log(), p_clean_i2t, reduction="batchmean", log_target=True)
    loss_trades_smooth_t2i = F.kl_div(p_smooth_t2i.log(), p_clean_t2i, reduction="batchmean", log_target=True)

    loss_cons_i2t = _kl_mean_pairwise(dists_i2t)
    loss_cons_t2i = _kl_mean_pairwise(dists_t2i)

    return {
        "trades_smooth_i2t": loss_trades_smooth_i2t,
        "trades_smooth_t2i": loss_trades_smooth_t2i,
        "cons_i2t": loss_cons_i2t,
        "cons_t2i": loss_cons_t2i,
        "sigma_mean_img": sig_img.mean().detach(),
        "sigma_mean_txt": sig_txt.mean().detach(),
        "Tprime_mean_i2t": T_i2t.mean().detach(),
        "Tprime_mean_t2i": T_t2i.mean().detach(),
    }

# ========= label-aware core T2T (not paraphrase-style) =========

def label_aware_t2t_core(
    txt_a: torch.Tensor,
    txt_b: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
    assume_normalized: bool = True,
) -> torch.Tensor:
    if not assume_normalized:
        txt_a = _l2norm(txt_a)
        txt_b = _l2norm(txt_b)

    logits = float(logit_scale) * (txt_a @ txt_b.t())
    B = logits.size(0)
    dev = logits.device
    eye = torch.eye(B, device=dev)
    same_label = (labels.view(B, 1) == labels.view(1, B)).float()
    pos_mask = torch.maximum(same_label, eye)
    masked_logits = logits + (1.0 - pos_mask) * (-1e4)
    targets = torch.arange(B, device=dev)
    return F.cross_entropy(masked_logits, targets)


__all__ = [
    "clip_like_loss",
    "text_only_contrastive_loss",
    "label_aware_clip_xmodal",
    "paraphrase_consensus_loss",
    "label_aware_paraphrase_consensus",
    "label_aware_i2i",
    "label_aware_t2t_core",
    "sbi_cross_label_loss",
    "image_consistency_loss",
    "temperature_reg",
    "cross_modal_trades",
    "flip_aware_margin",
    "trades_t2t",
    "trades_i2i",
    "robust_alignment_loss",
    "forgery_proto_loss",
    "mass_losses_light",
    "sim_matrix",
    "sim_logits",
]