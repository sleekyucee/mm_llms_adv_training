#scripts/utils_mm.py

import os
import yaml
import torch
import random
import numpy as np

_WIN1252_MAP = {
    "\x91": "'",
    "\x92": "'", 
    "\x93": '"',
    "\x94": '"',
    "\x96": "-",
    "\x97": "-",
    "\xa0": " ",
}

def _sanitize_text(s: str) -> str:
    return s.translate(str.maketrans(_WIN1252_MAP))

def load_config(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        txt = _sanitize_text(txt)
        return yaml.safe_load(txt)
    except UnicodeDecodeError:
        pass

    with open(path, "rb") as f:
        raw = f.read()

    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            txt = raw.decode(enc, errors="strict")
            txt = _sanitize_text(txt)
            cfg = yaml.safe_load(txt)
            print(f"[CFG] decoded {os.path.basename(path)} as {enc}", flush=True)
            return cfg
        except Exception:
            continue

    txt = raw.decode("utf-8", errors="replace")
    txt = _sanitize_text(txt)
    print(f"[CFG][WARN] used utf-8 with replacement characters for {os.path.basename(path)}", flush=True)
    return yaml.safe_load(txt)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True