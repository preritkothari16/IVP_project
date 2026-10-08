"""Re-split step 2: cluster-aware stratified train/val/test split.

Whole near-duplicate clusters are kept intact, so no frame of a plant/berry can appear in two
splits. Allocation is stratified per class and greedy toward the current per-class proportions
(about 70/20/10 train/val/test) so results stay comparable to the shipped model.

Written to dataset_v2/ using HARDLINKS, so no image data is duplicated and dataset/ stays
untouched as a rollback.
"""
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

DS = Path(r"D:\ivp\plant_ai\dataset")
V2 = Path(r"D:\ivp\plant_ai\dataset_v2")
SEED = 0
TARGETS = {"train": 0.70, "val": 0.20, "test": 0.10}

rows = [r for r in csv.DictReader((DS / "manifest.csv").open(encoding="utf-8"))
        if r["is_dup_copy"] != "True" and "_hard" not in r["path"]]
cl = json.loads(Path(r"D:\ivp\plant_ai\resplit_clusters.json").read_text(encoding="utf-8"))

path2row = {r["path"]: r for r in rows}
clusters = [set(v) for v in cl["clusters"].values()]
print(f"images {len(rows)}  clusters {len(clusters)}")

# --- cross-class clusters: a cluster spanning two labels is a hard conflict ---------
cross = 0
cross_imgs = 0
for c in clusters:
    cls = {path2row[p]["final_class"] for p in c if p in path2row}
    if len(cls) > 1:
        cross += 1
        cross_imgs += len(c)
print(f"clusters spanning >1 class: {cross}  (images {cross_imgs})")
if cross:
    for c in clusters:
        cls = {path2row[p]["final_class"] for p in c if p in path2row}
        if len(cls) > 1:
            print(f"    {sorted(cls)}  {[path2row[p]['path'] for p in sorted(c)][:4]}")

# Assign whole clusters to splits, stratified by the class of their members.
# For a cluster spanning classes, it is assigned as a unit and counts toward each member class.
assign = {}
rng = random.Random(SEED)
by_class_clusters = defaultdict(list)
for c in clusters:
    members = [p for p in c if p in path2row]
    if not members:
        continue
    primary = Counter(path2row[p]["final_class"] for p in members).most_common(1)[0][0]
    by_class_clusters[primary].append(members)
singles = [p for p in path2row if not any(p in c for c in clusters)]
for p in singles:
    by_class_clusters[path2row[p]["final_class"]].append([p])

rng.shuffle(singles)
for cls, cl_list in by_class_clusters.items():
    members_total = sum(len(m) for m in cl_list)
    # largest clusters first, random tie-break: putting a big cluster anywhere moves the needle
    # most, and doing it early keeps the tail from being unable to balance.
    cl_list.sort(key=lambda m: (-len(m), rng.random()))
    got = {s: 0 for s in TARGETS}
    placed = 0
    for members in cl_list:
        k = len(members)
        # Proportional rule: pick the split furthest behind its target *relative to progress*.
        # Absolute deficits are wrong here - test and val have small absolute targets, so they
        # would absorb everything and train would starve.
        best = max(TARGETS, key=lambda s: TARGETS[s] * placed - got[s])
        for p in members:
            assign[p] = best
        got[best] += k
        placed += k

# --- report -----------------------------------------------------------------------
new = defaultdict(Counter)
for p, s in assign.items():
    new[path2row[p]["final_class"]][s] += 1

print()
print(f"{'class':24s} {'n':>5s} {'train':>7s} {'val':>6s} {'test':>6s}   | {'was tr/va/te':>16s}")
print("-" * 82)
tot = Counter()
for cls in sorted(new):
    n = sum(new[cls].values())
    tot.update(new[cls])
    was = Counter(r["split"] for r in rows if r["final_class"] == cls)
    print(f"{cls:24s} {n:5d} {new[cls]['train']:7d} {new[cls]['val']:6d} {new[cls]['test']:6d}   |"
          f" {was['train']:5d}/{was['val']:4d}/{was['test']:4d}")
print("-" * 82)
N = sum(tot.values())
print(f"{'TOTAL':24s} {N:5d} {tot['train']:7d} {tot['val']:6d} {tot['test']:6d}")
print(f"{'pct':24s} {'':5s} {100*tot['train']/N:6.1f}% {100*tot['val']/N:5.1f}% {100*tot['test']/N:5.1f}%")

# --- verify no cluster straddles --------------------------------------------------
bad = 0
for c in clusters:
    sp = {assign[p] for p in c if p in assign}
    if len(sp) > 1:
        bad += 1
print(f"\nclusters straddling splits: {bad}  (must be 0)")

empty = [(cls, s) for cls in new for s in TARGETS if new[cls][s] == 0]
print(f"empty class/split cells  : {len(empty)} {empty if empty else ''}")

Path(r"D:\ivp\plant_ai\resplit_assign.json").write_text(json.dumps(assign, indent=0), encoding="utf-8")
print("\nwrote resplit_assign.json")