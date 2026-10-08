"""Re-split step 3: how independent is the new TEST set, really?

Important honesty check before training. This re-split re-partitions the SAME image pool, so it
removes within-split near-duplicate leakage but it does NOT create a genuinely untouched test
set: every image in it was available during earlier research. Quantify exactly how much moved.
"""
import csv
import json
from collections import Counter
from pathlib import Path

DS = Path(r"D:\ivp\plant_ai\dataset")
rows = [r for r in csv.DictReader((DS / "manifest.csv").open(encoding="utf-8"))
        if r["is_dup_copy"] != "True" and "_hard" not in r["path"]]
assign = json.loads(Path(r"D:\ivp\plant_ai\resplit_assign.json").read_text(encoding="utf-8"))
old = {r["path"]: r["split"] for r in rows}
cls = {r["path"]: r["final_class"] for r in rows}

mat = Counter()
for p, new_s in assign.items():
    mat[(old[p], new_s)] += 1
order = ["train", "val", "test"]
print("=" * 72)
print("MIGRATION MATRIX  rows = OLD split, cols = NEW split")
print("=" * 72)
print(f"{'':10s}" + "".join(f"{c:>12s}" for c in order) + f"{'total':>10s}")
for o in order:
    t = sum(mat[(o, n)] for n in order)
    print(f"{o:10s}" + "".join(f"{mat[(o,n)]:12d}" for n in order) + f"{t:10d}")
print(f"{'total':10s}" + "".join(f"{sum(mat[(o,n)] for o in order):12d}" for n in order)
      + f"{sum(mat.values()):10d}")

new_test = [p for p in assign if assign[p] == "test"]
old_test = [p for p in old if old[p] == "test"]
overlap = sum(1 for p in new_test if old[p] == "test")
from_train = sum(1 for p in new_test if old[p] == "train")
from_val = sum(1 for p in new_test if old[p] == "val")

print()
print(f"NEW test size              : {len(new_test)}")
print(f"  reused from OLD test     : {overlap}")
print(f"  newly withheld (was train): {from_train}")
print(f"  newly withheld (was val)  : {from_val}")

print()
print("per-class NEW test composition vs OLD test:")
print(f"{'class':24s} {'old':>5s} {'new':>5s} {'same images':>13s}")
for c in sorted({cls[p] for p in assign}):
    o = sum(1 for p in old_test if cls[p] == c)
    nt = [p for p in new_test if cls[p] == c]
    same = sum(1 for p in nt if old[p] == "test")
    print(f"{c:24s} {o:5d} {len(nt):5d} {same:13d}")

# what fraction of OLD test was in the same near-dup cluster as a train image?
cl_json = json.loads(Path(r"D:\ivp\plant_ai\resplit_clusters.json").read_text(encoding="utf-8"))
leaky_old, leaky_new = 0, 0
for members in cl_json["clusters"].values():
    ms = set(members)
    if any(old.get(p) == "test" for p in ms) and any(old.get(p) == "train" for p in ms):
        leaky_old += sum(1 for p in ms if old.get(p) == "test")
    if any(assign.get(p) == "test" for p in ms) and any(assign.get(p) == "train" for p in ms):
        leaky_new += sum(1 for p in ms if assign.get(p) == "test")
print()
print(f"test images sharing a near-dup cluster with a TRAIN image:")
print(f"  OLD split: {leaky_old} of {len(old_test)}  ({100*leaky_old/len(old_test):.1f}%)")
print(f"  NEW split: {leaky_new} of {len(new_test)}  ({100*leaky_new/len(new_test):.1f}%)")