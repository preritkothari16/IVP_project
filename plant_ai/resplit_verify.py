"""Re-split step 5: verify dataset_v2 before any training happens on it.

Three checks that must all pass:
  1. no near-duplicate cluster straddles a split boundary in dataset_v2
  2. val and test contain NO augmented (_hard) files
  3. every real image appears exactly once, and counts match the assignment
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DS = Path(r"D:\ivp\plant_ai\dataset")
V2 = Path(r"D:\ivp\plant_ai\dataset_v2")
assign = json.loads(Path(r"D:\ivp\plant_ai\resplit_assign.json").read_text(encoding="utf-8"))
clusters = json.loads(Path(r"D:\ivp\plant_ai\resplit_clusters.json").read_text(encoding="utf-8"))["clusters"]

# where did each original file land?  Augmented _hard variants are counted separately, because
# they legitimately sit in train and would otherwise look like extra real images.
landed = {}
aug = Counter()
for split in ("train", "val", "test"):
    for p in (V2 / split).rglob("*.jpg"):
        if "_hard" in p.stem:
            aug[split] += 1
            continue
        landed[p.name] = split

ok = True

# 1. cluster straddling
bad = []
for members in clusters.values():
    sp = {landed.get(Path(m).name) for m in members}
    if len({x for x in sp if x}) > 1:
        bad.append(members)
print(f"[1] clusters straddling splits in dataset_v2: {len(bad)}")
if bad:
    ok = False
    for m in bad[:5]:
        print("     ", [Path(x).name for x in m])

# 2. augmented files outside train
for split in ("val", "test"):
    n = sum(1 for p in (V2 / split).rglob("*_hard*.jpg"))
    print(f"[2] augmented *_hard* files in {split}: {n}")
    if n:
        ok = False

# 3. counts and completeness
exp = Counter(assign.values())
got = Counter(landed.values())
print(f"[3] real images per split  expected {dict(exp)}  got {dict(got)}")
print(f"    augmented variants: {dict(aug)}  (train-only by design)")
if exp != got:
    ok = False
    print("     MISMATCH")
if any(aug[s] for s in ("val", "test")):
    ok = False
missing = [p for p in assign if Path(p).name not in landed]
print(f"    real images missing from dataset_v2: {len(missing)}")
if missing:
    ok = False

# class presence
print()
print(f"{'class':24s} {'train':>7s} {'val':>6s} {'test':>6s}")
for c in sorted({r.split('/')[-2] for r in assign}):
    t = sum(1 for s, d in (("train", "train"), ("val", "val"), ("test", "test"))
            for p in (V2 / s / c).glob("*.jpg"))
    print(f"{c:24s} {t:7d}")

print()
print("VERDICT:", "PASS - safe to train" if ok else "FAIL - do not train")