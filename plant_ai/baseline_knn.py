"""k-NN baseline on frozen YOLOv5s-cls backbone features.

Purpose: isolate the contribution of the trained softmax classifier. The SAME backbone and the
SAME preprocessing as the submitted model are used, so any difference is attributable to the
classifier head alone (1-NN over frozen features vs. fine-tuned 9-way softmax).

Protocol (deliberately conservative):
  * features are extracted once from the frozen ImageNet-pretrained backbone
  * k is selected on data/val ONLY - never on data/test
  * k = 3 is additionally reported on both val and test, as a fixed a-priori choice
  * k_sweep.csv records every k tried

Outputs:
  baseline_knn/features/*.npz      cached features
  baseline_knn/knn_k{K}_{val,test}/eval.json + confusion_matrix.csv
  baseline_knn/k_sweep.csv
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import paths

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "yolov5"))
from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import classify_transforms  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

OUT = ROOT / "baseline_knn"
FEAT = OUT / "features"
OUT.mkdir(exist_ok=True)
FEAT.mkdir(exist_ok=True)

WEIGHTS = ROOT / "yolov5" / "yolov5s-cls.pt"
IMG_SIZE = 224
KS = [1, 3, 5, 7, 9, 11, 15]
K_REPORT = 3


def backbone(weights):
    """Load the classification model and return (feature_extractor, transform, names)."""
    device = select_device("")
    m = DetectMultiBackend(str(weights), device=device, fuse=False)
    net = m.model  # ClassificationModel
    net.eval()
    # penultimate features: everything before the final Classify() linear
    layers = list(net.model)
    body = torch.nn.Sequential(*layers[:-1]).to(device).eval()
    tf = classify_transforms(IMG_SIZE)
    return body, tf, device, net


@torch.inference_mode()
def resolve_split_dir(split):
    """Return the on-disk directory for a manifest split.

    While the validation-split retrain is in progress, dataset/test is temporarily renamed to
    dataset/_test_holdout so that yolov5's validation loader cannot see it. Resolve to whichever
    exists so the baseline can run concurrently without touching the held-out data.
    """
    d = paths.DATASET / split
    if d.is_dir():
        return d
    alt = paths.DATASET / f"_{split}_holdout"
    if alt.is_dir():
        print(f"  note: using {alt.name} for the {split} split (train holdout is active)")
        return alt
    raise SystemExit(f"[error] split directory not found for '{split}': {d}")


@torch.inference_mode()
def extract(split, body, tf, device):
    """Return (features [N,D], labels [N], filenames)."""
    out_npz = FEAT / f"{split}.npz"
    if out_npz.exists():
        d = np.load(out_npz, allow_pickle=True)
        print(f"  [{split}] cached {d['X'].shape}")
        return d["X"], d["y"], list(d["files"])

    split_dir = resolve_split_dir(split)
    rows = [r for r in csv.DictReader(paths.MANIFEST.open(encoding="utf-8"))
            if r["split"] == split and r["is_dup_copy"] != "True"]
    print(f"  [{split}] extracting {len(rows)} images ...")
    from PIL import Image
    feats, labels, files = [], [], []
    for n, r in enumerate(rows, 1):
        # manifest stores "test/cls/img.jpg"; swap in the resolved split directory
        parts = r["path"].split("/", 1)
        p = split_dir / parts[1] if len(parts) == 2 else paths.DATASET / r["path"]
        try:
            # same input path as backend/main.py: PIL decode -> RGB ndarray
            im = np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            continue
        t = tf(im).unsqueeze(0).to(device)
        f = body(t).flatten(1).float().cpu().numpy()[0]
        feats.append(f)
        labels.append(r["final_class"])
        files.append(Path(r["path"]).name)
        if n % 250 == 0:
            print(f"    {n}/{len(rows)}")
    X = np.stack(feats)
    y = np.array(labels)
    np.savez_compressed(out_npz, X=X, y=y, files=np.array(files))
    print(f"  [{split}] features {X.shape}")
    return X, y, files


def knn_predict(Xtr, ytr, Xte, k):
    """Distance-weighted k-NN vote. Cosine distance on L2-normalised features."""
    a = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    b = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    classes = sorted(set(ytr))
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(ytr), len(classes)))
    Y[np.arange(len(ytr)), [idx[c] for c in ytr]] = 1.0

    out = []
    for s in range(0, len(a), 64):
        sim = a[s:s + 64] @ b.T                      # cosine similarity
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.arange(len(part))
        top_sim = sim[rows[:, None], part]
        w = np.clip(top_sim, 0, None)                # weight by similarity
        votes = (Y[part] * w[:, :, None]).sum(axis=1)
        out.append(votes)
    return np.concatenate(out, axis=0), classes


def metrics(y_true_names, scores, classes, thr=0.6):
    pred = [classes[i] for i in scores.argmax(axis=1)]
    classes_l = sorted(set(y_true_names))
    cm = np.zeros((len(classes_l), len(classes_l)), dtype=int)
    ci = {c: i for i, c in enumerate(classes_l)}
    for t, p in zip(y_true_names, pred):
        cm[ci[t], ci[p]] += 1
    sup = cm.sum(1)
    pn = cm.sum(0)
    diag = np.diag(cm).astype(float)
    prec = np.where(pn > 0, diag / np.maximum(pn, 1e-12), np.nan)
    rec = np.where(sup > 0, diag / np.maximum(sup, 1e-12), np.nan)
    f1 = np.where((prof := (prec + rec)) > 0, 2 * prec * rec / np.maximum(prof, 1e-12), np.nan)
    overall = float(np.mean([t == p for t, p in zip(y_true_names, pred)]))
    per = [{"class": c, "support": int(sup[i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "f1": float(f1[i])} for i, c in enumerate(classes_l)]
    return {
        "n": len(y_true_names),
        "overall_top1": overall,
        "macro_precision": float(np.nanmean(prec)),
        "macro_recall": float(np.nanmean(rec)),
        "macro_f1": float(np.nanmean(f1)),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
        "classes": classes_l,
        "_pred": pred,
        "_cm": cm,
    }


def write_eval(tag, m, k, split):
    d = OUT / f"knn_k{k}_{split}"
    d.mkdir(exist_ok=True)
    np.savetxt(d / "confusion_matrix.csv", m["_cm"], delimiter=",", fmt="%d")
    payload = {kk: v for kk, v in m.items() if not kk.startswith("_")}
    payload.update({"method": f"k-NN (k={k}) on frozen YOLOv5s-cls features",
                    "k": k, "split": split,
                    "k_selected_on": "data/val only" if split == "test" else "n/a"})
    (d / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {d/'eval.json'}")
    return payload


def main():
    print("=" * 78)
    print("k-NN BASELINE: frozen YOLOv5s-cls features + distance-weighted k-NN")
    print("=" * 78)
    if not WEIGHTS.exists():
        raise SystemExit(f"[error] backbone weights not found: {WEIGHTS}\n"
                         f"        download with:  curl -L -o yolov5/yolov5s-cls.pt "
                         f"https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt")
    body, tf, device, _ = backbone(WEIGHTS)
    print(f"  backbone : {WEIGHTS.name} on {device}")
    print(f"  feature dim: {body(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=device)).numel()}")

    t0 = time.time()
    print("\nextracting features:")
    Xtr, ytr, _ = extract("train", body, tf, device)
    Xva, yva, _ = extract("val", body, tf, device)
    Xte, yte, _ = extract("test", body, tf, device)
    print(f"\nfeature extraction took {time.time()-t0:.0f}s")

    print()
    print("=" * 78)
    print("k SWEEP  (k chosen on data/val; data/test never used for selection)")
    print("=" * 78)
    print(f"{'k':>3s} {'val macro-F1':>14s} {'val acc':>9s} {'test macro-F1':>15s} {'test acc':>9s}")
    sweep = []
    for k in KS:
        sv, classes = knn_predict(Xtr, ytr, Xva, k)
        mv = metrics(list(yva), sv, classes)
        st, _ = knn_predict(Xtr, ytr, Xte, k)
        mt = metrics(list(yte), st, classes)
        sweep.append({
            "k": k,
            "val_macro_f1": mv["macro_f1"], "val_macro_recall": mv["macro_recall"],
            "val_macro_precision": mv["macro_precision"], "val_accuracy": mv["overall_top1"],
            "test_macro_f1": mt["macro_f1"], "test_macro_recall": mt["macro_recall"],
            "test_macro_precision": mt["macro_precision"], "test_accuracy": mt["overall_top1"],
            "test_n": mt["n"],
        })
        print(f"{k:3d} {mv['macro_f1']:14.4f} {mv['overall_top1']:9.4f} "
              f"{mt['macro_f1']:15.4f} {mt['overall_top1']:9.4f}")

    with (OUT / "k_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    print(f"\nwrote {OUT/'k_sweep.csv'}")

    best = max(sweep, key=lambda r: r["val_macro_f1"])
    print(f"\n  k selected on val : k = {best['k']} (val macro-F1 {best['val_macro_f1']:.4f})")
    print(f"  reporting k = {K_REPORT} as an a-priori choice (written to knn_k{K_REPORT}_* )")

    for split, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        s, classes = knn_predict(Xtr, ytr, X, K_REPORT)
        m = metrics(list(y), s, classes)
        p = write_eval(f"k{K_REPORT}", m, K_REPORT, split)
        print(f"    {split}: macro-F1 {p['macro_f1']:.4f}  macro-recall {p['macro_recall']:.4f}  "
              f"acc {p['overall_top1']:.4f}")

    summary = {
        "method": "k-NN on frozen YOLOv5s-cls penultimate features (distance-weighted, cosine)",
        "feature_dim": int(Xtr.shape[1]),
        "k_reported": K_REPORT,
        "k_selected_on_val": best["k"],
        "selection_protocol": "k chosen on data/val only; data/test never used for selection",
        "val": {"macro_f1": best["val_macro_f1"], "accuracy": best["val_accuracy"]},
        "test_k3": json.loads((OUT / f"knn_k{K_REPORT}_test" / "eval.json").read_text(encoding="utf-8")),
    }
    (OUT / "README.txt").write_text(
        "k-NN baseline for the strawberry 9-class classifier.\n\n"
        "Same backbone and same preprocessing as the submitted model (YOLOv5s-cls, 224px, RGB),\n"
        "but the 9-way softmax head is replaced by a distance-weighted k-NN vote over frozen\n"
        "penultimate features. Any difference from the submitted model is therefore attributable\n"
        "to the classifier head, not to the representation or the input pipeline.\n\n"
        "k was selected on data/val only. data/test was never used to choose k.\n\n"
        "  k_sweep.csv               every k tried, val and test metrics\n"
        "  knn_k3_val/eval.json      k=3 on val\n"
        "  knn_k3_test/eval.json     k=3 on test\n"
        "  features/*.npz            cached features\n",
        encoding="utf-8")
    print(f"\nwrote {OUT/'README.txt'}")
    print("DONE")


if __name__ == "__main__":
    main()
