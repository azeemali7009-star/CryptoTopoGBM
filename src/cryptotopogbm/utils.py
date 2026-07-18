from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Any
import numpy as np

def seed_everything(seed:int=7):
    random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    except Exception:
        pass

def ensure_dir(path: str|Path)->Path:
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p

def write_json(obj: dict[str,Any], path: str|Path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
