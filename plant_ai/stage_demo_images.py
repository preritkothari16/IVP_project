"""
Stage demo images at the project root.

- 1-2 correctly-predicted TEST images per class (named by class) so nothing needs hunting.
- leaf_spot_410.jpg  -> the low-confidence UI state.
- a powdery_mildew_fruit <-> gray_mold ambiguous image -> shows merged Powdery Mildew grouping.
Selection is deterministic (seeded) and every copy is verified correct against the saved
TEST predictions, so a "known good" really is known-good.
"""

import paths  # central path configuration; see paths.py
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = paths.PROJECT
YOLOV5 = PROJECT / "yolov5"
DATASET = paths.DATASET
DEMO = paths.DEMO_IMAGES
EVAL = paths.EVAL_DIR / "ep28_corrected"
SEED = 7
LOW_CONF_SRC = DATASET / "test" / "leaf_spot" / "leaf_spot_410.jpg"
WEIGHTS = PROJECT / "backend" / "best.pt"

sys.path.append(str(YOLOV5))
from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import classify_transforms  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


device = select_device("")
_model = DetectMultiBackend(str(WEIGHTS), device=device, fuse=False)
_raw = _model.names
_names = [str(_raw[i]) for i in sorted(_raw)] if isinstance(_raw, dict) else [str(n) for n in _raw]
_tf = classify_transforms(224)


def top1_conf(rel_path):
    im = cv2.imread(str(DATASET / rel_path))
    if im is None:
        return 0.0
    t = _tf(im).unsqueeze(0).to(device)
    with torch.inference_mode():
        return float(F.softmax(_model(t), dim=1)[0].max().item())

DEMO.mkdir(exist_ok=True)
for old in DEMO.glob("*.jpg"):
    old.unlink()

# ground truth from the TEST evaluation, not from filenames
preds = {r["path"]: r for r in csv.DictReader((EVAL / "predictions.csv").open(encoding="utf-8"))}
rows = [r for r in csv.DictReader((DATASET / "manifest.csv").open(encoding="utf-8"))
        if r["split"] == "test" and r["is_dup_copy"] == "False"]

correct = defaultdict(list)
for r in rows:
    p = preds.get(r["path"])
    if p and p["correct"] == "1":
        correct[r["final_class"]].append(r["path"])

rng = random.Random(SEED)
staged = []
for cls in sorted(correct):
    # score candidates and keep the most confidently-correct ones
    scored = sorted(((top1_conf(rel), rel) for rel in correct[cls]), key=lambda t: -t[0])
    take = 1  # exactly one known-good per class keeps the demo tight
    for conf, rel in scored[:take]:
        src = DATASET / rel
        stem = Path(rel).stem
        clean = stem.replace("Strawberry___", "")
        dst = DEMO / f"{cls}__{clean}.jpg"
        shutil.copy2(src, dst)
        staged.append((cls, dst.name, conf, stem))

# --- the two deliberate hard cases ---
if LOW_CONF_SRC.exists():
    shutil.copy2(LOW_CONF_SRC, DEMO / "HARD__low_confidence__leaf_spot_410.jpg")
    staged.append(("leaf_spot (LOW CONFIDENCE)", "HARD__low_confidence__leaf_spot_410.jpg", None, "leaf_spot_410"))

# a powdery_mildew_fruit image the model called gray_mold (documented label ambiguity)
ambig = None
for r in rows:
    if r["final_class"] == "powdery_mildew_fruit":
        p = preds.get(r["path"])
        if p and p["correct"] == "0" and p["pred"] == "gray_mold":
            ambig = r["path"]
            break
if ambig:
    src = DATASET / ambig
    stem = Path(ambig).stem
    dst = DEMO / f"HARD__ambiguous__{stem}.jpg"
    shutil.copy2(src, dst)
    staged.append(("powdery_mildew_fruit (AMBIGUOUS)", dst.name, None, stem))

print("=" * 78)
print("demo_images/ staged")
print("=" * 78)
print(f"{'class':34s} {'file':48s} {'conf':>7s}")
print("-" * 78)
for cls, name, conf, stem in staged:
    c = f"{conf:.3f}" if conf is not None else "n/a"
    print(f"{cls:34s} {name:48s} {c:>7s}")
print("-" * 78)
print(f"total files: {len(list(DEMO.glob('*.jpg')))}")
