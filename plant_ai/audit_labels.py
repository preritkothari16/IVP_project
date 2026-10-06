"""Step 1 of the label audit: identify each class's most-confused rival from the SHIPPED
model's confusion matrix (eval/ep100clean_val), plus the low-confidence rate as context.

This is label-error detection. Confidence is reported only as context - it is NOT used to
select candidates, because low-confidence images are frequently hard-but-correct and removing
them would degrade the training set.
"""
import csv
import json
from pathlib import Path

E = Path(r"D:\ivp\plant_ai\eval\ep100clean_val")
d = json.loads((E / "eval.json").read_text(encoding="utf-8"))
cm = json.loads(json.dumps(d["per_class"]))  # keep for ordering
import numpy as np

cmat = np.loadtxt(E / "confusion_matrix.csv", delimiter=",", dtype=int)
classes = [p["class"] for p in d["per_class"]]

preds = list(csv.DictReader((E / "predictions.csv").open(encoding="utf-8")))
conf_of = {r["path"]: float(r["confidence"]) for r in preds}
low = [r for r in preds if float(r["confidence"]) < 0.6]

print("=" * 96)
print("SHIPPED MODEL: confusion matrix (rows = true, cols = predicted)")
print("=" * 96)
print(f"{'':24s}" + "".join(f"{c[:9]:>10s}" for c in classes))
for i, c in enumerate(classes):
    print(f"{c:24s}" + "".join(f"{cmat[i,j]:10d}" for j in range(len(classes))))

print()
print("=" * 96)
print("PER-CLASS RIVAL = the class this one is most often confused WITH (either direction)")
print("=" * 96)
print(f"{'class':24s} {'n':>4s} {'acc':>7s} {'primary rival':24s} {'count':>6s} {'%':>6s}")
print("-" * 84)
rivals = {}
for i, c in enumerate(classes):
    n = cmat[i].sum()
    acc = cmat[i, i] / n if n else 0
    row = cmat[i].copy()
    row[i] = 0
    col = cmat[:, i].copy()
    col[i] = 0
    # combined bidirectional confusion
    combined = row + col
    j = int(np.argmax(combined))
    rivals[c] = classes[j]
    print(f"{c:24s} {n:4d} {acc:7.3f} {classes[j]:24s} {combined[j]:6d} {100*combined[j]/n:5.1f}%")

print()
print("=" * 96)
print("CONTEXT ONLY (not used for selection): low-confidence images per class")
print("=" * 96)
from collections import Counter
lc = Counter(r["true"] for r in low)
te = Counter(r["true"] for r in preds)
for c in classes:
    print(f"  {c:24s} {lc.get(c,0):3d} of {te[c]:3d} test images below 0.60  ({100*lc.get(c,0)/te[c]:5.1f}%)")

print()
print("=" * 96)
print("AUDIT PLAN: each class scored against its rival's train distribution")
print("=" * 96)
for c in classes:
    print(f"  {c:24s} -> rival {rivals[c]:24s} n_rival_train=?")
print()
print("Rival pairs to audit (rival may be an audit target itself):")
seen = set()
for c in classes:
    pair = tuple(sorted((c, rivals[c])))
    if pair not in seen:
        seen.add(pair)
        print(f"  {c} vs {rivals[c]}")