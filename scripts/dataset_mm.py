# scripts/dataset_mm.py

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _train_tf_clip(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.9, 1.0),
                ratio=(0.98, 1.02),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.15,
                        saturation=0.10,
                        hue=0.02,
                    )
                ],
                p=0.6,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.2,
            ),
            transforms.RandomGrayscale(p=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def _eval_tf_clip(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


class FFPPTripletDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        frames_root: str,
        image_size: int = 224,
        augment: bool = True,
        project_root: Optional[str] = None,
        clip_norm: bool = True,
        return_pair: bool = True,
    ) -> None:
        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

        self.frames_root = Path(frames_root).resolve()
        self.project_root = (
            Path(project_root).resolve() if project_root else Path.cwd().resolve()
        )
        self.return_pair = bool(return_pair)

        if clip_norm:
            tf_base = _train_tf_clip(image_size) if augment else _eval_tf_clip(image_size)
            self.tf = tf_base
            self.tf2 = _train_tf_clip(image_size) if augment else _eval_tf_clip(image_size)
        else:
            if augment:
                self.tf = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
            else:
                self.tf = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                    ]
                )
            self.tf2 = (
                transforms.Compose(self.tf.transforms) if self.return_pair else None
            )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_frame_path(self, rel_path: str) -> Path:
        return (self.frames_root / rel_path).resolve()

    def _resolve_blend_path(self, p: str) -> Path:
        p_path = Path(p)
        if p_path.is_absolute():
            return p_path
        return (self.project_root / p).resolve()

    @staticmethod
    def _load_img_pil(p: Path) -> Image.Image:
        return Image.open(p).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        img_path = self._resolve_frame_path(r["image_rel"])
        pil = self._load_img_pil(img_path)

        image = self.tf(pil)
        image2 = self.tf2(pil) if (self.return_pair and self.tf2 is not None) else None

        out: Dict[str, Any] = {
            "image": image,
            "image2": image2,
            "texts": r.get("texts", []),
            "label": int(r.get("label", 0)),
            "source_type": r.get("source_type", "original"),
            "blend_image": None,
        }

        blend_rel = r.get("blend_img", "")
        if blend_rel:
            try:
                blend_path = self._resolve_blend_path(blend_rel)
                if blend_path.exists():
                    pil_b = self._load_img_pil(blend_path)
                    out["blend_image"] = self.tf(pil_b)
            except Exception:
                out["blend_image"] = None

        if "image_rel" in r:
            out["image_rel"] = r["image_rel"]
        if "image_path" in r:
            out["image_path"] = r["image_path"]
        if "source_type" in r:
            out["source"] = r["source_type"]
        if "manip_type" in r:
            out["manip_type"] = r["manip_type"]
        if "manip" in r:
            out["manip"] = r["manip"]

        return out

