import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sentence_transformers import SentenceTransformer

try:
    import open_clip
except Exception:
    open_clip = None


class StyleAdapter(nn.Module):
    def __init__(self, d: int, k: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d + k, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )

    def forward(self, x, style_code):
        z = torch.cat([x, style_code], dim=-1)
        return self.fc(z)


class StyleCodeMaker(nn.Module):
    def __init__(self, k: int = 8):
        super().__init__()
        self.src_vocab = {
            "original": 0,
            "Deepfakes": 1,
            "FaceSwap": 2,
            "Face2Face": 3,
            "NeuralTextures": 4,
        }
        self.len_vocab = {"short": 0, "medium": 1, "long": 2}
        k_src = k // 2
        k_len = k - k_src
        self.src_emb = nn.Embedding(len(self.src_vocab), k_src)
        self.len_emb = nn.Embedding(len(self.len_vocab), k_len)
        nn.init.normal_(self.src_emb.weight, std=0.02)
        nn.init.normal_(self.len_emb.weight, std=0.02)

    def forward(self, source_types, length_bins, device=None):
        if device is None:
            device = next(self.parameters()).device
        src_idx = torch.tensor(
            [self.src_vocab.get(s, 0) for s in source_types],
            device=device,
            dtype=torch.long,
        )
        len_idx = torch.tensor(
            [self.len_vocab.get(b, "medium") for b in length_bins],
            device=device,
            dtype=torch.long,
        )
        code = torch.cat([self.src_emb(src_idx), self.len_emb(len_idx)], dim=-1)
        return code


class ForensicToken(nn.Module):
    """
    Fixed Sobel high-pass -> magnitude -> global average -> MLP to embed_dim.
    Only the projection MLP is learnable (tiny head).
    """
    def __init__(self, embed_dim: int, hidden_mult: float = 2.0):
        super().__init__()
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)
        hidden = int(hidden_mult * embed_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        gray = 0.2989 * imgs[:, 0:1] + 0.5870 * imgs[:, 1:2] + 0.1140 * imgs[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        g = mag.mean(dim=(2, 3), keepdim=True).view(mag.size(0), 1)
        return self.proj(g)


class FiLMRefiner(nn.Module):
    """
    Cross-modal FiLM in embedding space:
      - refine image with cond = [text || forensic]
      - optionally refine text with cond = [image || forensic]
    """
    def __init__(self, embed_dim: int, symmetric: bool = True, dropout: float = 0.05):
        super().__init__()
        self.symmetric = symmetric
        hid = 4 * embed_dim

        self.ln_img = nn.LayerNorm(embed_dim)
        self.ln_txt = nn.LayerNorm(embed_dim)
        self.ln_cond = nn.LayerNorm(2 * embed_dim)

        self.to_gamma_beta_img = nn.Sequential(
            nn.Linear(2 * embed_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 2 * embed_dim),
        )
        if symmetric:
            self.to_gamma_beta_txt = nn.Sequential(
                nn.Linear(2 * embed_dim, hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hid, 2 * embed_dim),
            )

        self.alpha_img = nn.Parameter(torch.tensor(0.5))
        self.alpha_txt = nn.Parameter(torch.tensor(0.5))

    def refine_img(self, img: torch.Tensor, txt: torch.Tensor, forensic: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([txt, forensic], dim=-1)
        gb = self.to_gamma_beta_img(self.ln_cond(cond))
        gamma, beta = gb.chunk(2, dim=-1)
        img_f = self.ln_img(img)
        out = img_f * (1 + gamma) + beta
        return img + self.alpha_img * out

    def refine_txt(self, img: torch.Tensor, txt: torch.Tensor, forensic: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([img, forensic], dim=-1)
        gb = self.to_gamma_beta_txt(self.ln_cond(cond))
        gamma, beta = gb.chunk(2, dim=-1)
        txt_f = self.ln_txt(txt)
        out = txt_f * (1 + gamma) + beta
        return txt + self.alpha_txt * out

    def forward(self, img: torch.Tensor, txt: torch.Tensor, forensic: torch.Tensor):
        img2 = self.refine_img(img, txt, forensic)
        if hasattr(self, "to_gamma_beta_txt"):
            txt2 = self.refine_txt(img, txt, forensic)
        else:
            txt2 = txt
        return img2, txt2


class MMModel(nn.Module):
    def __init__(
        self,
        img_dim: int = 512,
        txt_dim: int = 384,
        proj_dim: int = 256,
        mini_lm_path: str = "data/mini_lm_embedder",
        style_k: int = 8,
        backbone: str = "resnet50",
        pretrained: bool = True,
        model_type: str = "resnet_sbert",
        clip_model_name: str = "ViT-L-14",
        clip_pretrained_tag: str = None,
        clip_ckpt_path: str = None,
        novelty_cfg: dict = None,
    ):
        super().__init__()
        self.model_type = model_type
        self.novelty_cfg = novelty_cfg or {}

        if self.model_type == "clip":
            assert open_clip is not None, "open_clip is not installed in this environment."

            self.clip, _, _ = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=(clip_pretrained_tag if clip_ckpt_path is None else None)
            )

            if clip_ckpt_path:
                sd = torch.load(clip_ckpt_path, map_location="cpu")
                missing, unexpected = self.clip.load_state_dict(sd, strict=False)
                if (len(missing) + len(unexpected)) > 0:
                    print(f"[CLIP] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

            if bool(self.novelty_cfg.get("grad_checkpoint", False)):
                if hasattr(self.clip, "set_grad_checkpointing"):
                    self.clip.set_grad_checkpointing(enable=True)
                elif hasattr(self.clip, "transformer") and hasattr(self.clip.transformer, "set_grad_checkpointing"):
                    self.clip.transformer.set_grad_checkpointing(enable=True)

            self.logit_scale = self.clip.logit_scale

            self.embed_dim = getattr(self.clip, "embed_dim", None)
            if self.embed_dim is None:
                tp = getattr(self.clip, "text_projection", None)
                if tp is not None and hasattr(tp, "shape"):
                    self.embed_dim = int(tp.shape[-1])
                else:
                    self.embed_dim = 768

            ft_cfg = self.novelty_cfg.get("forensic_token", {})
            self.use_forensic = bool(ft_cfg.get("enable", False))
            if self.use_forensic:
                hidden_mult = float(ft_cfg.get("hidden_mult", 2.0))
                self.forensic_tok = ForensicToken(self.embed_dim, hidden_mult=hidden_mult)

            fr_cfg = self.novelty_cfg.get("film_refiner", {})
            self.use_refiner = bool(fr_cfg.get("enable", False))
            if self.use_refiner:
                symmetric = bool(fr_cfg.get("symmetric", True))
                dropout = float(fr_cfg.get("dropout", 0.05))
                self.film_refiner = FiLMRefiner(self.embed_dim, symmetric=symmetric, dropout=dropout)

            self.style_maker = None

        else:
            self.image = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
            self.img_proj = nn.Linear(self.image.num_features, proj_dim)

            self.text = SentenceTransformer(mini_lm_path)
            for p in self.text.parameters():
                p.requires_grad = False
            self.txt_proj = nn.Linear(txt_dim, proj_dim)

            self.adapter = StyleAdapter(d=txt_dim, k=style_k)
            self.style_maker = StyleCodeMaker(k=style_k)

            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
            self.embed_dim = proj_dim

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "clip":
            z = self.clip.encode_image(x)
            return F.normalize(z, dim=-1)
        else:
            z = self.image(x)
            z = self.img_proj(z)
            return F.normalize(z, dim=-1)

    def encode_text(self, texts, style_code=None, device="cpu") -> torch.Tensor:
        if self.model_type == "clip":
            toks = open_clip.tokenize(texts).to(device)
            z = self.clip.encode_text(toks)
            return F.normalize(z, dim=-1)
        else:
            with torch.no_grad():
                emb = self.text.encode(texts, convert_to_tensor=True, device=device)
            if style_code is not None:
                emb = self.adapter(emb, style_code)
            z = self.txt_proj(emb)
            return F.normalize(z, dim=-1)

    def make_style_code(self, source_types, length_bins, device=None):
        if self.model_type == "clip" or (self.style_maker is None):
            return None
        return self.style_maker(source_types, length_bins, device=device)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp()

    def set_backbone_trainable(self, flag: bool):
        if self.model_type == "clip":
            for p in self.clip.visual.parameters():
                p.requires_grad = flag
        else:
            for p in self.image.parameters():
                p.requires_grad = flag

    def refine_embeddings(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, imgs: torch.Tensor = None):
        if not getattr(self, "use_refiner", False):
            return img_emb, txt_emb

        if getattr(self, "use_forensic", False) and (imgs is not None):
            f = self.forensic_tok(imgs)
        else:
            f = torch.zeros_like(img_emb)

        img_f, txt_f = self.film_refiner(img_emb.float(), txt_emb.float(), f.float())
        img_f = F.normalize(img_f, dim=-1).to(img_emb.dtype)
        txt_f = F.normalize(txt_f, dim=-1).to(txt_emb.dtype)
        return img_f, txt_f