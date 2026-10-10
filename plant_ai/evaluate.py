"""
Evaluate a YOLOv5 classification checkpoint on the TEST split of D:\\ivp\\plant_ai\\dataset.

Reports macro-averaged top-1 accuracy, per-class accuracy, a confusion matrix image,
the most-confused class pairs, and accuracy split by PlantVillage-source vs other-source
test images (background-bias guard).

Usage: python evaluate.py <weights.pt> <out_tag> [--test-dir ...]
"""

import paths  # central path configuration; see paths.py
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Cap BLAS/OpenMP threads before torch is imported, or torch's CUDA DLLs fail to load with
# WinError 1455 on this many-core, memory-constrained machine. Must precede `import torch`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import threadcaps  # noqa: F401
except Exception:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = paths.PROJECT
YOLOV5 = PROJECT / "yolov5"
DATASET = paths.DATASET
IMG_SIZE = 224
# Centre-crop scales used for TTA: full frame plus progressively tighter crops.
TTA_VIEW_SCALES = [1.0, 0.85, 0.7, 0.55]

sys.path.append(str(YOLOV5))
from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import classify_transforms  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402



def load_manifest():
    rows = list(csv.DictReader((DATASET / "manifest.csv").open(encoding="utf-8")))
    return [r for r in rows if r["split"] == "test" and r["is_dup_copy"] == "False"]


def tta_views(im, n=8):
    """Deterministic test-time augmentation views of an RGB HWC array.

    TTA_VIEW_SCALES x {identity, hflip} = 8 views at the default n=8. Centre-cropping to
    several scales probes how much of the frame the model depends on, which is exactly the
    failure axis for the small fruit classes (berry small in a busy frame). Views are averaged
    in probability space.
    """
    h, w = im.shape[:2]
    scales = TTA_VIEW_SCALES[: max(1, n // 2)] if n >= 2 else [1.0]
    views = []
    for s in scales:
        ch, cw = max(1, int(h * s)), max(1, int(w * s))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = im[y0:y0 + ch, x0:x0 + cw]
        if crop.size == 0:
            crop = im
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        views.append(crop)
        if len(views) < n:
            views.append(cv2.flip(crop, 1))
    return views[:n] if n else [im]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("tag")
    ap.add_argument("--tta", type=int, default=0,
                    help="test-time augmentation: how many views to average (0=off). "
                         "Views are deterministic (hflip + centre crop at several scales); their "
                         "probabilities are averaged.")
    args = ap.parse_args()

    weights = Path(args.weights)
    outdir = PROJECT / "eval" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    device = select_device("")
    model = DetectMultiBackend(str(weights), device=device, fuse=False)
    raw = model.names
    names = [str(raw[i]) for i in sorted(raw)] if isinstance(raw, dict) else [str(n) for n in raw]
    tf = classify_transforms(IMG_SIZE)
    model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))
    print(f"[eval] weights = {weights}")
    print(f"[eval] classes = {names}")
    print(f"[eval] device  = {device}\n")

    rows = load_manifest()
    print(f"[eval] test images = {len(rows)}")
    if args.tta:
        print(f"[eval] TTA enabled: averaging {args.tta} deterministic views per image")

    y_true, y_pred, pv_flags, paths, confs = [], [], [], [], []
    for n, r in enumerate(rows, 1):
        p = DATASET / r["path"]
        im = cv2.imread(str(p))
        if im is None:
            print(f"  [skip unreadable] {r['path']}")
            continue
        # cv2.imread returns BGR; the model was trained on RGB (torchvision transforms).
        # BUGFIX: this conversion was missing, so evaluation ran on channel-swapped input.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        views = tta_views(im) if args.tta else [im]
        acc = np.zeros(len(names), dtype=np.float64)
        for v in views:
            t = tf(v).unsqueeze(0).to(device)
            t = t.half() if model.fp16 else t.float()
            with torch.inference_mode():
                acc += F.softmax(model(t), dim=1)[0].float().cpu().numpy()
        probs = acc / len(views)
        confs.append(float(probs.max()))
        y_true.append(names.index(r["final_class"]))
        y_pred.append(int(probs.argmax()))
        pv_flags.append(r["is_plantvillage"] == "True")
        paths.append(r["path"])
        if n % 50 == 0:
            print(f"  ...{n}/{len(rows)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pv = np.array(pv_flags)
    n_cls = len(names)

    # ---- overall, macro recall, macro F1 ----
    overall = float((y_true == y_pred).mean())

    # confusion matrix first: precision/recall/F1 all derive from it
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, q in zip(y_true, y_pred):
        cm[t, q] += 1
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm).astype(np.float64)

    per_class = []
    for i, nm in enumerate(names):
        n_i = int(support[i])
        tp = float(correct[i])
        prec = tp / predicted[i] if predicted[i] else float("nan")
        rec = tp / n_i if n_i else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if n_i and predicted[i] and (prec + rec) > 0 else float("nan")
        per_class.append({
            "class": nm, "support": n_i, "accuracy": rec,
            "precision": prec, "recall": rec, "f1": f1,
            "predicted": int(predicted[i]),
        })

    macro_recall = float(np.nanmean([p["recall"] for p in per_class]))
    macro_prec = float(np.nanmean([p["precision"] for p in per_class]))
    macro_f1 = float(np.nanmean([p["f1"] for p in per_class]))
    # majority-class baseline: always predict the most common true class
    majority = classes_support = support.max() / support.sum()
    majority_cls = names[int(np.argmax(support))]
    w = support / support.sum()
    weighted_f1 = float((w * np.nan_to_num([p["f1"] for p in per_class])).sum())

    print(f"\n{'=' * 78}")
    print(f"TEST  n={len(y_true)}")
    print(f"  overall top-1 (accuracy) : {overall:.4f}")
    print(f"  macro precision           : {macro_prec:.4f}")
    print(f"  macro recall (macro acc)  : {macro_recall:.4f}")
    print(f"  macro F1                  : {macro_f1:.4f}")
    print(f"  weighted F1               : {weighted_f1:.4f}")
    print(f"  majority-class baseline   : {majority:.4f}  (always predict '{majority_cls}')")
    print(f"{'=' * 78}")

    print(f"\n{'class':24s} {'n':>4s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'correct':>8s}")
    for p in sorted(per_class, key=lambda x: (np.isnan(x["f1"]), x["f1"])):
        c = int(round((p["recall"] if p["recall"] == p["recall"] else 0) * p["support"]))
        flag = "  <-- weakest" if p["f1"] == min(q["f1"] for q in per_class if q["support"]) else ""
        print(f"{p['class']:24s} {p['support']:4d} {p['precision']:7.4f} {p['recall']:7.4f} "
              f"{p['f1']:7.4f} {c:8d}{flag}")

    # ---- confusion matrix ----
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, q in zip(y_true, y_pred):
        cm[t, q] += 1

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im_ = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("predicted", fontsize=11)
    ax.set_ylabel("true", fontsize=11)
    ax.set_title(f"Confusion matrix - TEST (macro-F1 {macro_f1:.3f}, top-1 {overall:.3f})\n{weights.name}", fontsize=12)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(n_cls):
        for j in range(n_cls):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black", fontsize=9)
    fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_png = outdir / "confusion_matrix.png"
    fig.savefig(cm_png, dpi=140)
    plt.close(fig)
    print(f"\n[eval] confusion matrix -> {cm_png}")

    # ---- confused pairs ----
    pairs = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": names[i], "pred": names[j], "count": int(cm[i, j]),
                              "share_of_true": cm[i, j] / cm[i].sum() if cm[i].sum() else 0.0})
    pairs.sort(key=lambda d: -d["count"])
    print("\ntop 10 most-confused pairs (true -> predicted):")
    print(f"{'#':>3s} {'true':22s} {'predicted':22s} {'n':>4s} {'% of that class':>16s}")
    for k, d in enumerate(pairs[:10], 1):
        print(f"{k:3d} {d['true']:22s} {d['pred']:22s} {d['count']:4d} {d['share_of_true']:15.1%}")

    # ---- PlantVillage vs other ----
    print("\naccuracy by test-image source (background-bias guard):")
    seg = {}
    for label, mask in (("plantvillage", pv), ("other", ~pv)):
        n = int(mask.sum())
        a = float((y_true[mask] == y_pred[mask]).mean()) if n else float("nan")
        seg[label] = {"n": n, "accuracy": a}
        print(f"    {label:14s} n={n:4d}  top-1 = {a:.4f}")
    a_pv, a_ot = seg["plantvillage"]["accuracy"], seg["other"]["accuracy"]
    if not (np.isnan(a_pv) or np.isnan(a_ot)):
        print(f"    gap (plantvillage - other) = {a_pv - a_ot:+.4f}")
    # per-class PV vs other inside the classes that have both
    print("\n  per-class, where a class has both PV and other test images:")
    print(f"  {'class':24s} {'PV n':>5s} {'PV acc':>8s} {'oth n':>6s} {'oth acc':>8s}")
    mixed = []
    for i, nm in enumerate(names):
        m = y_true == i
        mp, mo = m & pv, m & ~pv
        if mp.sum() and mo.sum():
            ap_ = float((y_pred[mp] == i).mean())
            ao_ = float((y_pred[mo] == i).mean())
            mixed.append((nm, int(mp.sum()), ap_, int(mo.sum()), ao_))
            print(f"  {nm:24s} {int(mp.sum()):5d} {ap_:8.4f} {int(mo.sum()):6d} {ao_:8.4f}")
    if not mixed:
        print("  (none - each class appears with only one source in TEST)")

    # ---- persist ----
    out = {
        "weights": str(weights), "tag": args.tag, "n_test": len(y_true),
        "overall_top1": overall,
        # kept for backwards compatibility with earlier eval runs: this is macro RECALL
        # (macro-averaged per-class accuracy), not macro-F1. Prefer macro_f1 below.
        "macro_top1": macro_recall,
        "macro_precision": macro_prec,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "majority_class_baseline": {
            "class": majority_cls, "accuracy": float(majority),
            "note": "accuracy of always predicting the most frequent test class",
        },
        "target": {
            "value": 0.90,
            "metric": "macro_f1",
            "passed": bool(macro_f1 >= 0.90),
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
        },
        "per_class": per_class, "confused_pairs_top10": pairs[:10],
        "by_source": seg,
        "pv_other_gap": None if (np.isnan(a_pv) or np.isnan(a_ot)) else a_pv - a_ot,
        "classes_with_both_sources": [{"class": n, "pv_n": a, "pv_acc": b, "other_n": c, "other_acc": d}
                                      for n, a, b, c, d in mixed],
    }
    (outdir / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savetxt(outdir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    with (outdir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "true", "pred", "correct", "is_plantvillage", "confidence"])
        for pa, t, q, v, cf in zip(paths, y_true, y_pred, pv, confs):
            w.writerow([pa, names[t], names[q], int(t == q), bool(v), round(float(cf), 6)])
    print(f"\n[eval] wrote {outdir / 'eval.json'}, confusion_matrix.csv, predictions.csv")
    print(f"[eval] TARGET 90% macro-F1: {macro_f1:.4f} -> "
          f"{'PASS' if macro_f1 >= 0.90 else 'BELOW TARGET'}")
    print(f"[eval] (macro recall, the previously reported metric: {macro_recall:.4f} -> "
          f"{'PASS' if macro_recall >= 0.90 else 'BELOW TARGET'})")


if __name__ == "__main__":
    main()
