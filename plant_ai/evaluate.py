"""
Evaluate a YOLOv5 classification checkpoint on the TEST split of D:\\ivp\\plant_ai\\dataset.

Reports macro-averaged top-1 accuracy, per-class accuracy, a confusion matrix image,
the most-confused class pairs, and accuracy split by PlantVillage-source vs other-source
test images (background-bias guard).

Usage: python evaluate.py <weights.pt> <out_tag> [--test-dir ...]
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(r"D:\ivp\plant_ai")
YOLOV5 = PROJECT / "yolov5"
DATASET = PROJECT / "dataset"
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

    # ---- overall + macro ----
    overall = float((y_true == y_pred).mean())
    per_class = []
    for i, nm in enumerate(names):
        mask = y_true == i
        support = int(mask.sum())
        acc = float((y_pred[mask] == i).mean()) if support else float("nan")
        per_class.append({"class": nm, "support": support, "accuracy": acc})
    macro = float(np.nanmean([p["accuracy"] for p in per_class]))
    print(f"\n{'=' * 66}")
    print(f"TEST  n={len(y_true)}   overall top-1 = {overall:.4f}   MACRO top-1 = {macro:.4f}")
    print(f"{'=' * 66}")

    print("\nper-class accuracy:")
    print(f"{'class':24s} {'test n':>7s} {'acc':>8s} {'correct':>8s}")
    for p in sorted(per_class, key=lambda x: (np.isnan(x["accuracy"]), x["accuracy"])):
        c = int(round(p["accuracy"] * p["support"])) if p["support"] else 0
        flag = "  <-- weakest" if p["accuracy"] == min(q["accuracy"] for q in per_class if q["support"]) else ""
        print(f"{p['class']:24s} {p['support']:7d} {p['accuracy']:8.4f} {c:8d}{flag}")

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
    ax.set_title(f"Confusion matrix - TEST (macro {macro:.3f}, top-1 {overall:.3f})\n{weights.name}", fontsize=12)
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
        "overall_top1": overall, "macro_top1": macro,
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
    print(f"[eval] TARGET 90% macro: {'PASS' if macro >= 0.90 else 'BELOW TARGET'}")


if __name__ == "__main__":
    main()
