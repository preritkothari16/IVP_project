"""Re-split step 4: materialise dataset_v2/ with hardlinks.

dataset/ is left completely untouched so the shipped model remains reproducible and there is a
one-line rollback. Hardlinks mean no image bytes are duplicated.
"""
import csv
import json
import os
import shutil
from pathlib import Path

DS = Path(r"D:\ivp\plant_ai\dataset")
V2 = Path(r"D:\ivp\plant_ai\dataset_v2")

rows = [r for r in csv.DictReader((DS / "manifest.csv").open(encoding="utf-8"))
        if r["is_dup_copy"] != "True" and "_hard" not in r["path"]]
assign = json.loads(Path(r"D:\ivp\plant_ai\resplit_assign.json").read_text(encoding="utf-8"))

if V2.exists():
    shutil.rmtree(V2)

made = failed = 0
errors = []
for r in rows:
    p = r["path"]
    split = assign[p]
    dst = V2 / split / r["final_class"] / Path(p).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = DS / p
    if not src.exists():
        errors.append(f"missing source {src}")
        failed += 1
        continue
    try:
        os.link(src, dst)
        made += 1
    except OSError as e:
        # different volume or filesystem without link support: fall back to a copy
        shutil.copy2(src, dst)
        made += 1
        failed += 1
        errors.append(f"copied not linked: {src.name} ({e})")

print(f"hardlinked/created: {made}   fallbacks: {failed}")
for e in errors[:10]:
    print("  ", e)

# Hard-mined augmented variants are near-duplicates of their parent BY CONSTRUCTION, so they may
# only live in train, and only if their parent is also in train. If the parent landed in val or
# test, the variant cannot be used at all: keeping it in train would put a near-duplicate of a
# test image into the training set.
# Consequence: val and test contain REAL images only, and 24 variants whose parents moved to
# val/test are dropped, so training loses a little augmentation versus the shipped run.
n_aug = dropped_aug = 0
parent_split = {Path(r["path"]).stem: assign[r["path"]] for r in rows}

for r in csv.DictReader((DS / "manifest.csv").open(encoding="utf-8")):
    if r["is_dup_copy"] != "True":
        continue
    split = parent_split.get(r["dup_source"])
    if split is None:
        errors.append(f"no real parent for {r['path']} (dup_source={r['dup_source']})")
        continue
    if split != "train":
        dropped_aug += 1
        continue
    dst = V2 / split / r["final_class"] / Path(r["path"]).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = DS / r["path"]
    if src.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n_aug += 1
print(f"hard-mined variants in train: {n_aug}")
print(f"hard-mined variants DROPPED (parent in val/test, would leak): {dropped_aug}")
for e in errors[-5:]:
    print("  ERR", e)

# verify counts and that nothing is missing
for split in ("train", "val", "test"):
    d = V2 / split
    if not d.is_dir():
        continue
    tot = sum(1 for _ in d.rglob("*.jpg"))
    cls = sorted(x.name for x in d.iterdir() if x.is_dir())
    print(f"  {split:6s} {tot:5d} images, {len(cls)} classes")

tot_files = sum(1 for _ in V2.rglob("*.jpg"))
print(f"dataset_v2 total: {tot_files}")