"""For every misclassified TEST image, print the full probability vector so the
confusion can be judged from the model's own numbers."""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(r"D:\ivp\plant_ai")
YOLOV5 = PROJECT / "yolov5"
DATASET = PROJECT / "dataset"
WEIGHTS = YOLOV5 / "runs" / "train-cls" / "strawberry9" / "weights" / "best.pt"

sys.path.append(str(YOLOV5))
from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import classify_transforms  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

device = select_device("")
model = DetectMultiBackend(str(WEIGHTS), device=device, fuse=False)
raw = model.names
names = [str(raw[i]) for i in sorted(raw)] if isinstance(raw, dict) else [str(n) for n in raw]
tf = classify_transforms(224)

preds = {r["path"]: r for r in csv.DictReader((PROJECT / "eval" / "ep28" / "predictions.csv").open(encoding="utf-8"))}
rows = [r for r in csv.DictReader((DATASET / "manifest.csv").open(encoding="utf-8"))
        if r["split"] == "test" and r["is_dup_copy"] == "False"]

wrong = [r for r in rows if preds.get(r["path"], {}).get("correct") == "0"]
print(f"misclassified TEST images: {len(wrong)}\n")

for r in wrong:
    im = cv2.imread(str(DATASET / r["path"]))
    t = tf(im).unsqueeze(0).to(device)
    with torch.inference_mode():
        p = F.softmax(model(t), dim=1)[0]
    probs = p.detach().cpu().numpy()
    order = np.argsort(-probs)
    print(f"--- {r['path']}")
    print(f"    true = {r['final_class']:22s}  source = {r['source_folder']:26s}  PV = {r['is_plantvillage']}")
    top = names[order[0]]
    print(f"    pred = {top:22s}  (correct = {top == r['final_class']})")
    print("    probabilities:")
    for i in order[:4]:
        mark = " <-- TRUE" if names[i] == r["final_class"] else ""
        print(f"        {names[i]:24s} {probs[i]:.4f}{mark}")
    print(f"    margin (top1 - top2) = {probs[order[0]] - probs[order[1]]:.4f}\n")
