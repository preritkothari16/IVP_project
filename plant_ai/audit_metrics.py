"""Compute precision / recall / F1 from the existing confusion matrix.

evaluate.py currently reports accuracy only. "macro_top1" is the macro average of per-class
recall (accuracy), NOT macro-F1. For an imbalanced 9-class problem these differ materially, and
the "90% macro target" is ambiguous about which one it means. This quantifies the difference.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load(tag):
    cm = np.loadtxt(ROOT / "eval" / tag / "confusion_matrix.csv", delimiter=",")
    d = json.load((ROOT / "eval" / tag / "eval.json").open(encoding="utf-8"))
    return cm, d


for tag in ("ep100clean", "ep100clean_val", "ep100hard", "ep28_corrected"):
    cm, d = load(tag)
    classes = [p["class"] for p in d["per_class"]]
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)

    precision = np.where(predicted > 0, correct / np.maximum(predicted, 1e-12), 0.0)
    recall = np.where(support > 0, correct / np.maximum(support, 1e-12), 0.0)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)

    print("=" * 104)
    print(f"{tag}   n={d['n_test']}")
    print("=" * 104)
    print(f"{'class':24s} {'n':>4s} {'precision':>10s} {'recall':>8s} {'F1':>7s}")
    print("-" * 104)
    for i, c in enumerate(classes):
        print(f"{c:24s} {int(support[i]):4d} {precision[i]:10.4f} {recall[i]:8.4f} {f1[i]:7.4f}")
    print("-" * 104)
    print(f"{'MACRO (unweighted)':24s} {'':4s} {precision.mean():10.4f} {recall.mean():8.4f} {f1.mean():7.4f}")
    w = support / support.sum()
    print(f"{'WEIGHTED':24s} {'':4s} {(w*precision).sum():10.4f} {(w*recall).sum():8.4f} {(w*f1).sum():7.4f}")
    print()
    print(f"  reported 'macro_top1' (macro recall) : {d['macro_top1']:.4f}")
    print(f"  macro-F1                               : {f1.mean():.4f}")
    print(f"  difference                            : {d['macro_top1'] - f1.mean():+.4f}")
    maj = support.max() / support.sum()
    print(f"  majority-class baseline (accuracy)     : {maj:.4f}  "
          f"(always predict '{classes[int(np.argmax(support))]}')")
    print(f"  macro-F1 of majority baseline         : {np.nanmean(np.where(support == support.max(), 1.0, 0.0)):.4f}")
    print()
